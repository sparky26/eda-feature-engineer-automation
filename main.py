import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from groq import Groq  # Ensure GROQ_API_KEY is set

sns.set(style="whitegrid")

# ---------------------------
# Helpers / Preprocessing
# ---------------------------

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns and nested Python objects to strings/JSON for Streamlit."""
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == "object":
            df_safe[col] = df_safe[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict, set))
                else str(x) if x is not None else "missing"
            )
    return df_safe

def safe_display_df(df: pd.DataFrame):
    """Display dataframe in Streamlit after making it Arrow-safe."""
    st.dataframe(make_arrow_safe(df))

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess any CSV for robust EDA."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            # Try convert to numeric
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() / len(df) > 0.8:
                df[col] = numeric_col
            else:
                # Try datetime
                try:
                    dt_col = pd.to_datetime(df[col], errors='coerce')
                    if dt_col.notna().sum() / len(df) > 0.8:
                        df[col] = dt_col
                    else:
                        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict, set)) else str(x))
                except Exception:
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict, set)) else str(x))
    return df

# ---------------------------
# Robust EDA
# ---------------------------
# ---------------------------
# Robust EDA
# ---------------------------
def robust_eda(df: pd.DataFrame, target: str, max_plots=50, pairplot_sample=500):
    st.header("📊 Robust Exploratory Data Analysis")

    # Basic info
    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Column types:")
    st.write(df.dtypes)

    # Preview
    st.subheader("Preview (first 10 rows)")
    safe_display_df(df.head(10))

    # Missing values
    st.subheader("Missing Values")
    missing = df.isna().sum()
    missing_percent = 100 * missing / len(df)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_percent})
    safe_display_df(missing_df[missing_df["count"] > 0])

    # Numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.subheader("Numeric Feature Summary")
        safe_display_df(df[num_cols].describe())

        # Distributions
        st.subheader("Numeric Distributions")
        for col in num_cols[:max_plots]:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

        # Correlation heatmap
        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(min(12, 0.4*len(num_cols)+4), min(10, 0.4*len(num_cols)+4)))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Pairplot (sampled)
        if len(num_cols) > 1 and pairplot_sample > 0:
            st.subheader("Pairplot (Sampled Numeric Features)")
            sample_n = min(len(df), pairplot_sample)
            sample = df[num_cols].sample(sample_n, random_state=42)
            try:
                pair = sns.pairplot(sample)
                st.pyplot(pair.fig)
            except Exception as e:
                st.warning(f"Pairplot failed: {e}")

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.subheader("Categorical Feature Summary")
        cat_summary = pd.DataFrame({c: [df[c].nunique(), df[c].mode()[0] if not df[c].mode().empty else None] 
                                    for c in cat_cols}, index=["unique", "top"]).T
        safe_display_df(cat_summary)

        st.subheader("Categorical Distributions")
        for col in cat_cols[:max_plots]:
            top_vals = df[col].value_counts().head(50)
            fig, ax = plt.subplots()
            sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # Target-focused analysis
    if target in df.columns:
        st.subheader(f"Target Variable Analysis: {target}")
        target_dtype = df[target].dtype
        fig, ax = plt.subplots(figsize=(8, 4))
        if target_dtype in ["object", "category"]:
            sns.countplot(y=target, data=df, order=df[target].value_counts().index[:50], ax=ax)
        else:
            sns.histplot(df[target], kde=True, ax=ax)
        st.pyplot(fig)

        # Numeric features vs target
        if num_cols:
            st.subheader("Numeric Features vs Target")
            for col in num_cols[:max_plots]:
                fig, ax = plt.subplots(figsize=(8, 4))
                if target_dtype in ["object", "category"]:
                    sns.boxplot(x=target, y=col, data=df.sample(min(2000, len(df)), random_state=42), ax=ax)
                else:
                    sns.scatterplot(x=col, y=target, data=df.sample(min(2000, len(df)), random_state=42), ax=ax)
                ax.set_title(f"{col} vs {target}")
                st.pyplot(fig)

        # Categorical features vs target
        if cat_cols:
            st.subheader("Categorical Features vs Target")
            for col in cat_cols[:max_plots]:
                if col == target:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4))
                if target_dtype in ["object", "category"]:
                    sns.countplot(y=col, hue=target, data=df, order=df[col].value_counts().index[:50], ax=ax)
                else:
                    sns.boxplot(x=col, y=target, data=df.sample(min(2000, len(df)), random_state=42), ax=ax)
                ax.set_title(f"{col} vs {target}")
                st.pyplot(fig)

# ---------------------------
# Groq (AI) Suggestions
# ---------------------------
def ask_groq_for_recommendations(features: list, target: str, scenario: str) -> str:
    from dotenv import load_dotenv
    import os
    load_dotenv()  # Loads variables from .env into environment
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    features_text = ", ".join(features)
    prompt = f"""
Scenario: {scenario}

Dependent column to predict: {target}
Features to use: {features_text}

Act as a machine learning engineer expert and suggest:
1) Additional engineered features
2) Columns that may be problematic

Return a clear numbered plan.
"""
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq call failed: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("Exploratory Data Analyser+ Feature Engineering Assistant")

    st.markdown("""
    **How to use**:
    1. Upload CSV.  
    2. Select the dependent/target column.  
    3. Optionally select features. Leave blank to use all except target.  
    4. Click buttons to run EDA or Groq recommendations.
    """)

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

    max_plots = st.slider("Max features to plot", min_value=5, max_value=200, value=30)
    pairplot_sample = st.number_input("Pairplot sample size", min_value=0, max_value=2000, value=500)

    df_clean = preprocess_dataframe(df_raw)

    # User selects dependent column
    target = st.selectbox("Select dependent (target) column", df_clean.columns)

    # Feature selection
    features_text = st.text_input(
        f"Feature columns (comma-separated). Leave blank to use all except target `{target}`.",
        value=""
    )
    features = [f.strip() for f in features_text.split(",") if f.strip()] if features_text.strip() else [c for c in df_clean.columns if c != target]

    # Buttons
    col1, col2 = st.columns(2)
    run_eda_btn = col1.button("Run EDA")
    run_groq_btn = col2.button("Ask Feature Engine for Recommendations")

    if run_eda_btn:
        robust_eda(df_clean, target, max_plots=max_plots, pairplot_sample=pairplot_sample)

    if run_groq_btn:
        scenario_text = st.text_area("Scenario description for recommendation engine", "")
        if not scenario_text.strip():
            st.warning("Please provide scenario text for recommendations.")
        else:
            with st.spinner("Contacting Groq..."):
                analysis = ask_groq_for_recommendations(features, target, scenario_text)
                st.subheader("Feature ENgineering Analysis")
                st.markdown(analysis, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
