# data_processor.py - Data preprocessing and cleaning utilities

import json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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


def preprocess_dataframe(
    df: pd.DataFrame,
    impute_numeric: str = "median",   # "mean", "median", "none"
    impute_categorical: bool = True,
    scale_numeric = None,        # "standard", "minmax", None
    cap_outliers: bool = True,
    drop_constant: bool = True,
    drop_high_corr: float = 0.95
) -> pd.DataFrame:
    """Comprehensive data preprocessing pipeline."""
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