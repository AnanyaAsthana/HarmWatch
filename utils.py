# utils.py
# Helper functions used by the Streamlit app
# Fixed to handle duplicate column names

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

sns.set_style("whitegrid")


@st.cache_data
def load_data(uploaded_file):
    """Load CSV from uploaded file-like object or path. Tries common encodings."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    # 🔑 Ensure column names are unique to avoid DuplicateError
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df


def show_overview(df):
    st.subheader("Overview & Summary")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing cells", int(df.isnull().sum().sum()))

    st.write("**Data types**")
    dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
    dtypes["non_null_count"] = df.notnull().sum()
    st.dataframe(dtypes)

    st.write("**Descriptive statistics (numerical)**")
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        st.dataframe(num.describe().T)
    else:
        st.write("No numerical columns detected.")


def show_missing(df):
    st.subheader("Missing Values")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.write("No missing values detected.")
        return
    st.bar_chart(miss)

    st.write("Rows with missing values (top 10)")
    st.dataframe(df[df.isnull().any(axis=1)].head(10))


def show_distribution(df):
    st.subheader("Numerical Distributions")
    num = df.select_dtypes(include=[np.number]).copy()

    # 🔑 Remove duplicate columns before plotting
    num = num.loc[:, ~num.columns.duplicated()].copy()

    if num.empty:
        st.write("No numerical columns to plot.")
        return

    col = st.selectbox("Choose numerical column for histogram", options=list(num.columns), key="hist_col")
    bins = st.slider("Bins", 5, 200, 30, key="hist_bins")
    fig = px.histogram(num, x=col, nbins=bins, marginal="box", title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

    if len(num.columns) >= 2:
        st.write("Scatter plot between two numerical columns")
        c1, c2 = st.columns(2)
        with c1:
            xcol = st.selectbox("X column", options=num.columns, index=0, key="scatter_x")
        with c2:
            ycol = st.selectbox("Y column", options=num.columns, index=1, key="scatter_y")
        fig2 = px.scatter(num, x=xcol, y=ycol, trendline="ols", title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig2, use_container_width=True)


def show_correlation(df):
    st.subheader("Correlation (numerical)")
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, ~num.columns.duplicated()].copy()  # remove duplicates
    if num.shape[1] < 2:
        st.write("Need at least two numerical columns for correlation.")
        return
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    st.pyplot(fig)


def show_time_series(df):
    st.subheader("Time-series explorer")
    # detect datetime-like columns
    datetime_cols = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            datetime_cols.append(c)
        else:
            try:
                sample = df[c].dropna().astype(str).iloc[:20]
                pd.to_datetime(sample, errors="raise")
                datetime_cols.append(c)
            except Exception:
                pass

    if len(datetime_cols) == 0:
        st.write("No datetime-like columns detected.")
        return

    dt_col = st.selectbox("Choose datetime column", options=datetime_cols, key="dt_col")
    df_dt = df.copy()
    df_dt[dt_col] = pd.to_datetime(df_dt[dt_col], errors="coerce")
    df_dt = df_dt.dropna(subset=[dt_col])
    df_dt = df_dt.sort_values(dt_col)

    numeric_cols = df_dt.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.write("No numeric columns to plot against datetime.")
        return

    val_col = st.selectbox("Choose value column", options=numeric_cols, key="ts_val")
    window = st.slider("Rolling window (periods)", 1, 100, 7, key="ts_window")

    fig = px.line(df_dt, x=dt_col, y=val_col, title=f"{val_col} over time")
    if df_dt[val_col].notnull().sum() >= window:
        rolling = df_dt[val_col].rolling(window, min_periods=1).mean()
        fig.add_traces(px.line(df_dt, x=dt_col, y=rolling, labels={"y": f"{val_col} (rolling mean)"}).data)
    st.plotly_chart(fig, use_container_width=True)


def show_categorical(df):
    st.subheader("Categorical columns")
    cat = df.select_dtypes(include=["object", "category"]).copy()
    if cat.empty:
        st.write("No categorical columns detected.")
        return

    col = st.selectbox("Choose categorical column", options=list(cat.columns), key="cat_col")
    top_n = st.slider("Show top N categories", 3, 50, 10, key="cat_topn")
    counts = cat[col].value_counts().nlargest(top_n)
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": col, "y": "count"}, title=f"Top {top_n} categories in {col}")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Sample rows for each top category")
    samples = []
    for ix in counts.index:
        row = df[df[col] == ix].head(1)
        samples.append(row)
    if samples:
        st.dataframe(pd.concat(samples).reset_index(drop=True))


def show_text_analysis(df):
    st.subheader("Simple Text Analysis")
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(text_cols) == 0:
        st.write("No text columns detected.")
        return

    col = st.selectbox("Choose text column for analysis", options=text_cols, key="text_col")
    sample_text = " ".join(df[col].dropna().astype(str).head(100).tolist())
    if not sample_text.strip():
        st.write("Selected column contains no textual data.")
        return

    st.write("Wordcloud (top words)")
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(sample_text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Basic token frequency
    words = [w.lower().strip(".,!?:;()[]\"'") for w in sample_text.split() if len(w) > 2]
    c = Counter(words)
    top = c.most_common(20)
    freq_df = pd.DataFrame(top, columns=["word", "count"])
    fig2 = px.bar(freq_df, x="word", y="count", title="Top words")
    st.plotly_chart(fig2, use_container_width=True)

