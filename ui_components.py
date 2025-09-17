# ui_components.py - Streamlit UI components and interface logic

import streamlit as st
import pandas as pd
from data_processor import safe_display_df, preprocess_dataframe
from eda_visualizer import robust_eda
from ai_service import (
    ask_groq_for_recommendations, 
    parse_groq_table, 
    apply_feature_sketches,
    ask_groq_for_drops, 
    parse_drop_table
)


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(page_title="EDA & Auto-Feature Engineering", layout="wide")
    st.title("Exploratory Data Analysis & Auto-Feature Engineer")
    
    st.markdown("""
    Welcome!  
    This app helps you preprocess datasets, explore them visually, and leverage **Groq AI** for automated feature engineering and feature drop recommendations.
    """)


def setup_sidebar_options():
    """Setup preprocessing options in sidebar."""
    st.sidebar.header("Preprocessing Options")
    
    with st.sidebar.expander("Missing Values"):
        impute_numeric = st.selectbox("Numeric Imputation", ["median", "mean", "none"], index=0)
        impute_categorical = st.checkbox("Impute categoricals with 'missing'", value=True)

    with st.sidebar.expander("Scaling & Outliers"):
        scale_numeric = st.selectbox("Scale Numeric Features", ["None", "standard", "minmax"], index=0)
        scale_numeric_val = None if scale_numeric == "None" else scale_numeric
        cap_outliers = st.checkbox("Cap Outliers (1st–99th percentile)", value=True)

    with st.sidebar.expander("Feature Cleaning"):
        drop_constant = st.checkbox("Drop Constant Columns", value=True)
        drop_high_corr = st.slider("Drop Features with Correlation Above", 0.8, 0.99, 0.95, 0.01)
    
    return {
        'impute_numeric': impute_numeric,
        'impute_categorical': impute_categorical,
        'scale_numeric_val': scale_numeric_val,
        'cap_outliers': cap_outliers,
        'drop_constant': drop_constant,
        'drop_high_corr': drop_high_corr
    }


def handle_file_upload():
    """Handle CSV file upload and return dataframe."""
    uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if not uploaded:
        st.info("⬆Upload a CSV file to begin.")
        return None

    try:
        df_raw = pd.read_csv(uploaded)
        return df_raw
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return None


def create_main_tabs(df_raw, df_clean):
    """Create and handle the main application tabs."""
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Preprocessed Data", "EDA", "Feature Engineering"])

    with tab1:
        st.subheader("Raw Data Preview")
        safe_display_df(df_raw.head())

    with tab2:
        st.subheader("Preprocessed Data Preview")
        safe_display_df(df_clean.head())

    with tab3:
        st.subheader("Exploratory Data Analysis")
        target = st.selectbox("Select prediction variable", df_clean.columns)
        if st.button("Run EDA"):
            max_plots = st.slider("Max features to plot", 5, 200, 90)
            pairplot_sample = st.number_input("Pairplot sample size", 0, 2000, 500)
            robust_eda(df_clean, target, max_plots=max_plots, pairplot_sample=pairplot_sample)

    with tab4:
        handle_feature_engineering_tab(df_clean)


def handle_feature_engineering_tab(df_clean):
    """Handle the feature engineering tab interface."""
    st.subheader("Feature Engineering Assistant")
    target = st.selectbox("Prediction variable for modeling", df_clean.columns, key="target_groq")

    features_text = st.text_input("Optional: Specify features (comma-separated)", "")
    features = [f.strip() for f in features_text.split(",") if f.strip()] \
        if features_text.strip() else [c for c in df_clean.columns if c != target]

    scenario_text = st.text_area("📖 Describe the modeling scenario", "")

    if st.button("Run Feature Recommendations"):
        if not scenario_text.strip():
            st.warning("Please provide scenario text for engineered feature recommendations.")
        else:
            handle_ai_recommendations(df_clean, features, target, scenario_text)


def handle_ai_recommendations(df_clean, features, target, scenario_text):
    """Handle AI-powered feature recommendations."""
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