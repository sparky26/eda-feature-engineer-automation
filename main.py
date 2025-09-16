# main.py

import os
import re
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
    """
    Ask Groq for recommendations in a strict tabular format:
    | feature | sketch_of_creation_pandas | why_it_helps |
    """
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    features_text = ", ".join(features)
    prompt = f"""
Scenario: {scenario}

Dependent column to predict: {target}
Features to use: {features_text}

You are an expert feature engineer.

Your task is to suggest **new features** in a very strict tabular format with exactly three columns:

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
    """
    Parse Groq's markdown table into a pandas DataFrame with
    columns: feature, sketch_of_creation_pandas, why_it_helps
    """
    # Extract rows that look like table rows
    lines = [line.strip() for line in response.split("\n") if "|" in line and not line.startswith("|-")]
    rows = [line.split("|")[1:-1] for line in lines]  # remove first and last empty cells
    cleaned = [[c.strip() for c in row] for row in rows]

    if not cleaned or len(cleaned) < 2:
        raise ValueError("Groq response did not return a valid table.")

    df_table = pd.DataFrame(cleaned[1:], columns=cleaned[0])  # first row = header
    return df_table


def apply_feature_sketches(df: pd.DataFrame, table_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the pandas code sketches from Groq to transform the dataset.
    """
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


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.title("Exploratory Data Analysis & Auto-Feature Engineer")

    st.markdown("""
    **How to use**:
    1. Upload CSV.  
    2. Configure preprocessing options in the sidebar.  
    3. Select the dependent/target column.  
    4. Optionally select features. Leave blank to use all except target.  
    5. Click buttons to run EDA or Feature Engineering recommendations.
    """)

    # --- File Upload ---
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to begin.")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.subheader("Raw Preview")
    safe_display_df(df_raw.head())

    # ==========================================================
    # Sidebar Preprocessing Options
    # ==========================================================
    st.sidebar.header("⚙️ Preprocessing Options")

    impute_numeric = st.sidebar.selectbox(
        "Numeric Imputation Strategy", ["median", "mean", "none"], index=0
    )

    impute_categorical = st.sidebar.checkbox(
        "Impute Missing Categoricals with 'missing'", value=True
    )

    scale_numeric = st.sidebar.selectbox(
        "Scale Numeric Features", ["None", "standard", "minmax"], index=0
    )
    scale_numeric = None if scale_numeric == "None" else scale_numeric

    cap_outliers = st.sidebar.checkbox("Cap Outliers (1st–99th percentile)", value=True)

    drop_constant = st.sidebar.checkbox("Drop Constant Columns", value=True)

    drop_high_corr = st.sidebar.slider(
        "Drop Numeric Features with Correlation Above",
        min_value=0.8, max_value=0.99, value=0.95, step=0.01,
    )

    # ==========================================================
    # Apply Preprocessing
    # ==========================================================
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

    # ==========================================================
    # Target Column Selection
    # ==========================================================
    target = st.selectbox("Select dependent (target) column", df_clean.columns)

    # Feature selection
    features_text = st.text_input(
        f"Feature columns (comma-separated). Leave blank to use all except target `{target}`.",
        value=""
    )
    features = [f.strip() for f in features_text.split(",") if f.strip()] \
        if features_text.strip() else [c for c in df_clean.columns if c != target]

    # ==========================================================
    # Buttons
    # ==========================================================
    col1, col2 = st.columns(2)
    run_eda_btn = col1.button("Run EDA")
    run_groq_btn = col2.button("Run Feature Engineering Recommendation Engine")

    if run_eda_btn:
        max_plots = st.slider("Max features to plot", min_value=5, max_value=200, value=30)
        pairplot_sample = st.number_input("Pairplot sample size", min_value=0, max_value=2000, value=500)
        robust_eda(df_clean, target, max_plots=max_plots, pairplot_sample=pairplot_sample)

    if run_groq_btn:
        scenario_text = st.text_area("Scenario description for Groq AI", "")
        if not scenario_text.strip():
            st.warning("Please provide scenario text for Groq recommendations.")
        else:
            with st.spinner("Contacting Groq..."):
                analysis = ask_groq_for_recommendations(features, target, scenario_text)
                st.subheader("🤖 Groq AI Recommendations (Raw)")
                st.markdown(analysis)

                # Parse & apply
                try:
                    feature_table = parse_groq_table(analysis)
                    st.subheader("📑 Parsed Recommendations")
                    st.dataframe(feature_table)

                    transformed_df = apply_feature_sketches(df_clean, feature_table)

                    st.subheader("📊 Transformed Dataset Preview")
                    st.dataframe(transformed_df.head())

                    # Download
                    csv = transformed_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Transformed Dataset",
                        data=csv,
                        file_name="transformed_dataset.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Failed to parse or apply Groq output: {e}")


if __name__ == "__main__":
    main()
