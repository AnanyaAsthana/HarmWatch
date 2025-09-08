# app.py
# Streamlit entrypoint for social CSV analysis dashboard
# Run with: streamlit run app.py

import streamlit as st
from utils import (
    load_data,
    show_overview,
    show_missing,
    show_distribution,
    show_correlation,
    show_time_series,
    show_categorical,
    show_text_analysis,
)

st.set_page_config(page_title="Social Data Explorer", layout="wide")
st.title("Social Data Explorer 🧭")
st.markdown(
    "Upload a CSV and the app will run an exploratory analysis: preview, summary stats, missing values, distributions, correlations, time-series and simple text analysis."
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # load dataframe (handles file-like objects)
    df = load_data(uploaded_file)

    st.sidebar.header("Controls")
    show_head = st.sidebar.checkbox("Show head (first 10 rows)", value=True)
    if show_head:
        st.subheader("Data preview")
        st.dataframe(df.head(10))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Select columns for analysis (optional)")
    cols = st.sidebar.multiselect("Columns to focus on", options=list(df.columns), default=list(df.columns))
    if not cols:
        st.warning("Please select at least one column from the sidebar to proceed.")
    else:
        df_sel = df[cols].copy()

        # Overview
        show_overview(df_sel)

        # Missing values
        show_missing(df_sel)

        # Numerical distributions
        show_distribution(df_sel)

        # Correlation
        show_correlation(df_sel)

        # Time-series (auto-detect)
        show_time_series(df_sel)

        # Categorical
        show_categorical(df_sel)

        # Text analysis (if text exists)
        show_text_analysis(df_sel)

    st.sidebar.markdown("---")
    st.sidebar.info("Analysis generated automatically. For custom analyses, modify utils.py and app.py as needed.")

else:
    st.info("Upload a CSV file to begin. If your repo already contains a CSV (e.g., social_data.csv), drag it into the uploader or use the existing path when testing locally.")

