# eda_visualizer.py - Exploratory Data Analysis and visualization functions

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from data_processor import safe_display_df

def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename duplicate columns to be unique by appending _1, _2, etc."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            cols[idx] = f"{dup}_{i+1}"
    df.columns = cols
    return df

def robust_eda(
    df: pd.DataFrame, 
    target: str, 
    max_plots=50, 
    pairplot_sample=500, 
    max_features_per_pairplot=8, 
    max_features_per_heatmap=20
):
    """Comprehensive exploratory data analysis with interactive visualizations."""
    st.header("📊 Exploratory Data Analysis")

    # Original df for display, df_plot for plotting
    df_display = df.copy()
    df_plot = make_unique_columns(df.copy())

    # -------------------------
    # Dataset Overview
    # -------------------------
    with st.expander("📝 Dataset Overview", expanded=True):
        st.write(f"Rows: {df_display.shape[0]}, Columns: {df_display.shape[1]}")
        st.subheader("Column Types")
        dtypes_df = pd.DataFrame({
            'Column': df_display.columns,
            'Data Type': df_display.dtypes.astype(str)
        })
        safe_display_df(dtypes_df)
        st.subheader("Preview (first 10 rows)")
        safe_display_df(df_display.head(10))
        st.subheader("Missing Values")
        missing = df_display.isna().sum()
        missing_percent = 100 * missing / len(df_display)
        missing_df = pd.DataFrame({"count": missing, "percent": missing_percent})
        safe_display_df(missing_df[missing_df["count"] > 0])

    # -------------------------
    # Numeric Columns
    # -------------------------
    num_cols = df_display.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        with st.expander(f"🔢 Numeric Feature Summary ({len(num_cols)})", expanded=False):
            safe_display_df(df_display[num_cols].describe())

        with st.expander("📈 Numeric Distributions", expanded=False):
            for col in num_cols[:max_plots]:
                clean_data = df_display[col].dropna()
                if len(clean_data) == 0:
                    st.warning(f"No valid data for column {col}")
                    continue

                sample_size = min(len(clean_data), 5000)
                sample_data = clean_data.sample(sample_size, random_state=42)

                fig = px.histogram(
                    x=sample_data,
                    title=f"Distribution of {col}",
                    nbins=50,
                    marginal="box",
                    template="plotly_white"
                )
                fig.update_layout(xaxis_title=col, yaxis_title="Count", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            if len(num_cols) > max_plots:
                st.info(f"Skipped {len(num_cols) - max_plots} numeric columns.")

        with st.expander("🗂️ Correlation Heatmaps", expanded=False):
            for i in range(0, len(num_cols), max_features_per_heatmap):
                subset = num_cols[i:i+max_features_per_heatmap]
                if len(subset) > 1:
                    corr = df_display[subset].corr()
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title=f"Correlation Heatmap: {subset[0]} - {subset[-1]}",
                        template="plotly_white"
                    )
                    fig.update_layout(height=400, width=max(600, len(subset)*60))
                    fig.update_traces(
                        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔹 Interactive Scatter Matrix (Sampled)", expanded=False):
            sample_n = min(len(df_plot), pairplot_sample)
            sample = df_plot[num_cols].sample(sample_n, random_state=42)
            for i in range(0, len(num_cols), max_features_per_pairplot):
                subset = num_cols[i:i+max_features_per_pairplot]
                if len(subset) > 1:
                    try:
                        fig = px.scatter_matrix(
                            sample[subset],
                            dimensions=subset,
                            title=f"Scatter Matrix: {subset[0]} - {subset[-1]}",
                            template="plotly_white",
                            opacity=0.6
                        )
                        fig.update_layout(height=600, dragmode='select', hovermode='closest')
                        fig.update_traces(diagonal_visible=False, showupperhalf=False)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Scatter matrix failed for {subset}: {e}")

    # -------------------------
    # Categorical Columns
    # -------------------------
    cat_cols = df_display.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        with st.expander(f"🔠 Categorical Feature Summary ({len(cat_cols)})", expanded=False):
            cat_summary = pd.DataFrame({
                c: [df_display[c].nunique(), df_display[c].mode()[0] if not df_display[c].mode().empty else None] 
                for c in cat_cols
            }, index=["unique", "top"]).T
            safe_display_df(cat_summary)

        with st.expander("📊 Categorical Distributions", expanded=False):
            for col in cat_cols[:max_plots]:
                top_vals = df_display[col].value_counts().head(50)
                fig = px.bar(
                    x=top_vals.values,
                    y=top_vals.index,
                    orientation='h',
                    title=f"Distribution of {col}",
                    labels={'x': 'Count', 'y': col},
                    template="plotly_white"
                )
                fig.update_layout(height=max(400, len(top_vals)*20), yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            if len(cat_cols) > max_plots:
                st.info(f"Skipped {len(cat_cols) - max_plots} categorical columns.")

    # -------------------------
    # Target-focused Analysis
    # -------------------------
    if target in df_display.columns:
        target_dtype = df_display[target].dtype
        with st.expander(f"🎯 Target Variable Analysis: {target}", expanded=True):
            if np.issubdtype(target_dtype, np.number):
                fig = px.histogram(df_plot, x=target, title=f"Distribution of Target: {target}", nbins=50, marginal="box", template="plotly_white")
            else:
                fig = px.bar(df_plot[target].value_counts().reset_index(), x='index', y=target, title=f"Distribution of Target: {target}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # Categorical vs Target (numeric vs categorical plots)
        if cat_cols:
            with st.expander("🔹 Categorical Features vs Target", expanded=False):
                for col in cat_cols[:max_plots]:
                    if col == target:
                        continue
                    if np.issubdtype(df_display[target].dtype, np.number):
                        # numeric target with categorical feature: boxplot
                        fig = px.box(df_plot, x=col, y=target, title=f"{col} vs {target}", template="plotly_white")
                    else:
                        # categorical vs categorical
                        crosstab = pd.crosstab(df_display[col], df_display[target]).reset_index()
                        fig = px.bar(crosstab, x=col, y=crosstab.columns[1:], title=f"{col} vs {target}", template="plotly_white", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)
