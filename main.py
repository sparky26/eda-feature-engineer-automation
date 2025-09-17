# main.py (Improved UI/UX)

import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from preprocessing import preprocess_dataframe, robust_eda, safe_display_df
from groq import Groq

# ---------------------------
# Groq (AI) Suggestions
# ---------------------------

def ask_groq_for_recommendations(features: list, target: str, scenario: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    features_text = ", ".join(features)
    prompt = f"""
Scenario: {scenario}

Dependent column to predict: {target}
Features to use: {features_text}

You are an expert feature engineer.

Your task is to suggest **new features** in a strict tabular format with exactly three columns:

| feature | sketch_of_creation_pandas | why_it_helps |
|---------|----------------------------|--------------|
| NewFeatureName | df['NewFeatureName'] = df['col1'] / df['col2'] | Helps normalize values |
| FeatureBinned | df['FeatureBinned'] = pd.cut(df['col3'], bins=5, labels=False) | Reduces noise by binning continuous values |

⚠️ Rules:
- Only use valid **Python + Pandas one-liners** in `sketch_of_creation_pandas`.
- Do not include prose outside the table.
- If no new features are relevant, return an empty table with headers.
"""
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=4096,
            top_p=1,
            reasoning_effort="medium",
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq call failed: {e}"


def parse_groq_table(response: str) -> pd.DataFrame:
    lines = [line.strip() for line in response.split("\n") if "|" in line and not line.startswith("|-")]
    rows = [line.split("|")[1:-1] for line in lines]
    cleaned = [[c.strip() for c in row] for row in rows]

    if not cleaned or len(cleaned) < 2:
        raise ValueError("Groq response did not return a valid table.")

    return pd.DataFrame(cleaned[1:], columns=cleaned[0])


def apply_feature_sketches(df: pd.DataFrame, table_df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    for _, row in table_df.iterrows():
        feature = row["feature"]
        code = row["sketch_of_creation_pandas"]
        try:
            exec(code, {"df": df_new, "pd": pd, "np": np})
            st.success(f"✅ Created feature: {feature}")
        except Exception as e:
            st.warning(f"⚠️ Failed to create {feature}: {e}")
    return df_new


def ask_groq_for_drops(features: list, target: str, scenario: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    features_text = ", ".join(features)
    prompt = f"""
Scenario: {scenario}

Dependent column to predict: {target}
Candidate features: {features_text}

You are an expert ML practitioner.

Your task is to suggest **which features should be DROPPED** before modeling.
Return a strict markdown table with exactly two columns:

| feature | reason_for_dropping |
|---------|----------------------|
| colA | Too many missing values |
| colB | Highly correlated with target leakage |

⚠️ Rules:
- Only list features to be dropped.
- If none, return an empty table with just headers.
"""
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="medium",
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq call failed: {e}"


def parse_drop_table(response: str) -> pd.DataFrame:
    lines = [line.strip() for line in response.split("\n") if "|" in line and not line.startswith("|-")]
    rows = [line.split("|")[1:-1] for line in lines]
    cleaned = [[c.strip() for c in row] for row in rows]

    if not cleaned or len(cleaned) < 2:
        return pd.DataFrame(columns=["feature", "reason_for_dropping"])

    return pd.DataFrame(cleaned[1:], columns=cleaned[0])


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="EDA & Auto-Feature Engineering", layout="wide")
    st.title("Exploratory Data Analysis & Auto-Feature Engineer")

    st.markdown("""
    Welcome!  
    This app helps you preprocess datasets, explore them visually, and leverage **Groq AI** for automated feature engineering and feature drop recommendations.
    """)

    # --- File Upload ---
    uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if not uploaded:
        st.info("⬆Upload a CSV file to begin.")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return

    # Sidebar Options
    st.sidebar.header("Preprocessing Options")
    with st.sidebar.expander("Missing Values"):
        impute_numeric = st.selectbox("Numeric Imputation", ["median", "mean", "none"], index=0)
        impute_categorical = st.checkbox("Impute categoricals with 'missing'", value=True)

    with st.sidebar.expander("Scaling & Outliers"):
        scale_numeric = st.selectbox("Scale Numeric Features", ["None", "standard", "minmax"], index=0)
        scale_numeric = None if scale_numeric == "None" else scale_numeric
        cap_outliers = st.checkbox("Cap Outliers (1st–99th percentile)", value=True)

    with st.sidebar.expander("Feature Cleaning"):
        drop_constant = st.checkbox("Drop Constant Columns", value=True)
        drop_high_corr = st.slider("Drop Features with Correlation Above", 0.8, 0.99, 0.95, 0.01)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Preprocessed Data", "EDA", "Feature Engineering"])

    with tab1:
        st.subheader("Raw Data Preview")
        safe_display_df(df_raw.head())

    with tab2:
        st.subheader("Preprocessed Data Preview")
        df_clean = preprocess_dataframe(
            df_raw,
            impute_numeric=impute_numeric,
            impute_categorical=impute_categorical,
            scale_numeric=scale_numeric,
            cap_outliers=cap_outliers,
            drop_constant=drop_constant,
            drop_high_corr=drop_high_corr,
        )
        safe_display_df(df_clean.head())

    with tab3:
        st.subheader("Exploratory Data Analysis")
        target = st.selectbox("Select prediction variable", df_clean.columns)
        if st.button("Run EDA"):
            max_plots = st.slider("Max features to plot", 5, 200, 90)
            pairplot_sample = st.number_input("Pairplot sample size", 0, 2000, 500)
            robust_eda(df_clean, target, max_plots=max_plots, pairplot_sample=pairplot_sample)

    with tab4:
        st.subheader("Feature Engineering Assitant")
        target = st.selectbox("Prediction variable for modeling", df_clean.columns, key="target_groq")

        features_text = st.text_input("Optional: Specify features (comma-separated)", "")
        features = [f.strip() for f in features_text.split(",") if f.strip()] \
            if features_text.strip() else [c for c in df_clean.columns if c != target]

        scenario_text = st.text_area("📖 Describe the modeling scenario", "")

        if st.button("Run Feature Recommendations"):
            if not scenario_text.strip():
                st.warning("Please provide scenario text for engineered feature recommendations.")
            else:
                with st.spinner("🤖 Contacting Groq..."):
                    analysis = ask_groq_for_recommendations(features, target, scenario_text)
                st.markdown("### Feature Analysis and Implementation")
                st.markdown(analysis)

                try:
                    feature_table = parse_groq_table(analysis)
                    st.markdown("### Parsed Recommendations")
                    st.dataframe(feature_table)

                    transformed_df = apply_feature_sketches(df_clean, feature_table)
                    st.markdown("### Transformed Dataset Preview")
                    st.dataframe(transformed_df.head())

                    csv = transformed_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Transformed Dataset", data=csv,
                                       file_name="transformed_dataset.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"❌ Failed to parse or apply Groq output: {e}")

                st.markdown("---")
                st.markdown("### 🧹 Drop Feature Recommendations")
                with st.spinner("Asking Engine which features to drop..."):
                    drop_analysis = ask_groq_for_drops(features, target, scenario_text)
                st.markdown(drop_analysis)

                try:
                    drop_table = parse_drop_table(drop_analysis)
                    st.dataframe(drop_table)

                    if not drop_table.empty:
                        dropped = ", ".join(drop_table["feature"].tolist())
                        st.success(f"Groq recommends dropping: {dropped}")
                    else:
                        st.info("✅ No features suggested for dropping.")
                except Exception as e:
                    st.error(f"❌ Failed to parse Groq drop table: {e}")


if __name__ == "__main__":
    main()