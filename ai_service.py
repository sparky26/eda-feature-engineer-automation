# ai_service.py - AI-powered feature engineering recommendations

import os
import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq


def ask_groq_for_recommendations(features: list, target: str, scenario: str) -> str:
    """Ask Groq AI for feature engineering recommendations."""
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
    """Parse Groq response table into a pandas DataFrame."""
    lines = [line.strip() for line in response.split("\n") if "|" in line and not line.startswith("|-")]
    rows = [line.split("|")[1:-1] for line in lines]
    cleaned = [[c.strip() for c in row] for row in rows]

    if not cleaned or len(cleaned) < 2:
        raise ValueError("Groq response did not return a valid table.")

    return pd.DataFrame(cleaned[1:], columns=cleaned[0])


def apply_feature_sketches(df: pd.DataFrame, table_df: pd.DataFrame, auto_apply: bool = True) -> pd.DataFrame:
    """
    Apply feature engineering sketches to the dataframe.

    Parameters:
    - df: Original dataframe
    - table_df: Groq recommendations as a DataFrame
    - auto_apply: If True, apply all features automatically; if False, show buttons per feature

    Returns:
    - Transformed dataframe with new features applied
    """
    # Initialize session state for dataframe persistence
    if "df_new" not in st.session_state:
        st.session_state.df_new = df.copy()

    df_new = st.session_state.df_new

    if table_df.empty:
        st.info("✅ No feature recommendations to apply.")
        return df_new

    st.markdown("### ⚠️ Security Notice")
    st.warning("The AI has generated code that will be executed. Review before applying.")

    for _, row in table_df.iterrows():
        feature = row["feature"]
        code = row["sketch_of_creation_pandas"]

        with st.expander(f"Feature preview: {feature}", expanded=False):
            st.code(code, language="python")
            st.write(f"**Purpose:** {row.get('why_it_helps', 'No description provided')}")

        try:
            if auto_apply:
                # Automatically execute feature creation code
                safe_globals = {
                    "__builtins__": {},
                    "df": df_new,
                    "pd": pd,
                    "np": np
                }
                exec(code, safe_globals)
                st.success(f"✅ Created feature: {feature}")
            else:
                # Show button per feature
                if st.button(f"Apply {feature}", key=f"apply_{feature}"):
                    safe_globals = {
                        "__builtins__": {},
                        "df": df_new,
                        "pd": pd,
                        "np": np
                    }
                    exec(code, safe_globals)
                    st.success(f"✅ Created feature: {feature}")

        except Exception as e:
            st.error(f"❌ Failed to create {feature}: {e}")

    # Persist updated dataframe in session state
    st.session_state.df_new = df_new

    return df_new

def ask_groq_for_drops(features: list, target: str, scenario: str) -> str:
    """Ask Groq AI which features should be dropped."""
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
    """Parse Groq drop recommendations into a pandas DataFrame."""
    lines = [line.strip() for line in response.split("\n") if "|" in line and not line.startswith("|-")]
    rows = [line.split("|")[1:-1] for line in lines]
    cleaned = [[c.strip() for c in row] for row in rows]

    if not cleaned or len(cleaned) < 2:
        return pd.DataFrame(columns=["feature", "reason_for_dropping"])

    return pd.DataFrame(cleaned[1:], columns=cleaned[0])