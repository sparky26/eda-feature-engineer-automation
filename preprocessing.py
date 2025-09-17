# preprocessing.py (Interactive + Fixed correlation bug)

import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sns.set(style="whitegrid")

# ==========================================================
# Data Display Helpers
# ==========================================================
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

# ==========================================================
# Preprocessing
# ==========================================================
def preprocess_dataframe(
    df: pd.DataFrame,
    impute_numeric: str = "median",   # "mean", "median", "none"
    impute_categorical: bool = True,
    scale_numeric: str = None,        # "standard", "minmax", None
    cap_outliers: bool = True,
    drop_constant: bool = True,
    drop_high_corr: float = 0.95
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # ---- Type inference ----
    for col in df.columns:
        if df[col].dtype == "object":
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            if numeric_col.notna().sum() / len(df) > 0.8:
                df[col] = numeric_col
                continue
            dt_col = pd.to_datetime(df[col], errors="coerce")
            if dt_col.notna().sum() / len(df) > 0.8:
                df[col] = dt_col
                continue

    # ---- Handle missing values ----
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if impute_numeric != "none" and num_cols:
        strategy = "median" if impute_numeric == "median" else "mean"
        imputer = SimpleImputer(strategy=strategy)
        imputed = imputer.fit_transform(df[num_cols])
        df[num_cols] = pd.DataFrame(imputed, columns=num_cols, index=df.index)

    if impute_categorical and cat_cols:
        df[cat_cols] = df[cat_cols].fillna("missing")

    # ---- Cap outliers ----
    if cap_outliers:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # recompute after imputation
        for col in num_cols:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = np.clip(df[col], lower, upper)

    # ---- Scale numeric ----
    if scale_numeric:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scaler = StandardScaler() if scale_numeric == "standard" else MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

    # ---- Drop constant ----
    if drop_constant:
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)

    # ---- Drop highly correlated ----
    if drop_high_corr:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # recompute after drops
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > drop_high_corr)]
            if to_drop:
                df.drop(columns=to_drop, inplace=True)

    return df

# ==========================================================
# Robust EDA (Interactive)
# ==========================================================
def robust_eda(
    df: pd.DataFrame, 
    target: str, 
    max_plots=50, 
    pairplot_sample=500, 
    max_features_per_pairplot=8, 
    max_features_per_heatmap=20
):
    st.header("📊 Exploratory Data Analysis")

    # -------------------------
    # Dataset Overview
    # -------------------------
    with st.expander("📝 Dataset Overview", expanded=True):
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Column types:")
        st.write(df.dtypes)
        st.subheader("Preview (first 10 rows)")
        safe_display_df(df.head(10))
        st.subheader("Missing Values")
        missing = df.isna().sum()
        missing_percent = 100 * missing / len(df)
        missing_df = pd.DataFrame({"count": missing, "percent": missing_percent})
        safe_display_df(missing_df[missing_df["count"] > 0])

    # -------------------------
    # Numeric Columns
    # -------------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        with st.expander(f"🔢 Numeric Feature Summary ({len(num_cols)})", expanded=False):
            safe_display_df(df[num_cols].describe())

        with st.expander("📈 Numeric Distributions", expanded=False):
            for col in num_cols[:max_plots]:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna().sample(min(len(df), 5000)), kde=True, ax=ax)
                ax.set_title(col)
                st.pyplot(fig)
            if len(num_cols) > max_plots:
                st.info(f"Skipped {len(num_cols) - max_plots} numeric columns.")

        with st.expander("🗂️ Correlation Heatmaps", expanded=False):
            for i in range(0, len(num_cols), max_features_per_heatmap):
                subset = num_cols[i:i+max_features_per_heatmap]
                if len(subset) > 1:
                    corr = df[subset].corr()
                    fig, ax = plt.subplots(figsize=(min(12, 0.4*len(subset)+4), 8))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    ax.set_title(f"Heatmap: {subset[0]} - {subset[-1]}")
                    st.pyplot(fig)

        with st.expander("🔹 Pairplots (Sampled)", expanded=False):
            sample_n = min(len(df), pairplot_sample)
            sample = df[num_cols].sample(sample_n, random_state=42)
            for i in range(0, len(num_cols), max_features_per_pairplot):
                subset = num_cols[i:i+max_features_per_pairplot]
                if len(subset) > 1:
                    try:
                        pair = sns.pairplot(sample[subset])
                        st.pyplot(pair.fig)
                    except Exception as e:
                        st.warning(f"Pairplot failed for {subset}: {e}")

    # -------------------------
    # Categorical Columns
    # -------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        with st.expander(f"🔠 Categorical Feature Summary ({len(cat_cols)})", expanded=False):
            cat_summary = pd.DataFrame({
                c: [df[c].nunique(), df[c].mode()[0] if not df[c].mode().empty else None] 
                for c in cat_cols
            }, index=["unique", "top"]).T
            safe_display_df(cat_summary)

        with st.expander("📊 Categorical Distributions", expanded=False):
            for col in cat_cols[:max_plots]:
                top_vals = df[col].value_counts().head(50)
                fig, ax = plt.subplots()
                sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax)
                ax.set_title(col)
                st.pyplot(fig)
            if len(cat_cols) > max_plots:
                st.info(f"Skipped {len(cat_cols) - max_plots} categorical columns.")

    # -------------------------
    # Target-focused Analysis
    # -------------------------
    if target in df.columns:
        target_dtype = df[target].dtype
        with st.expander(f"🎯 Target Variable Analysis: {target}", expanded=True):
            fig, ax = plt.subplots(figsize=(8, 4))
            if target_dtype in ["object", "category"]:
                sns.countplot(y=target, data=df, order=df[target].value_counts().index[:50], ax=ax)
            else:
                sns.histplot(df[target], kde=True, ax=ax)
            st.pyplot(fig)

        if num_cols:
            with st.expander("📈 Numeric Features vs Target", expanded=False):
                for col in num_cols[:max_plots]:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    if target_dtype in ["object", "category"]:
                        sns.boxplot(x=target, y=col, data=df.sample(min(2000, len(df)), random_state=42), ax=ax)
                    else:
                        sns.scatterplot(x=col, y=target, data=df.sample(min(2000, len(df)), random_state=42), ax=ax)
                    ax.set_title(f"{col} vs {target}")
                    st.pyplot(fig)
                if len(num_cols) > max_plots:
                    st.info(f"Skipped {len(num_cols) - max_plots} numeric vs target plots.")

        if cat_cols:
            with st.expander("🔹 Categorical Features vs Target", expanded=False):
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
                if len(cat_cols) > max_plots:
                    st.info(f"Skipped {len(cat_cols) - max_plots} categorical vs target plots.")
