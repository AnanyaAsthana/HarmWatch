import streamlit as st
import pandas as pd
from utils import (
    analyze_echo_chambers,
    analyze_polarization,
    analyze_algorithmic_bias,
    analyze_misinformation,
    analyze_network_structure,
    plot_sentiment_distribution,
    plot_engagement_by_category,
    plot_temporal_content_spread,
    plot_user_content_diversity,
    plot_category_distribution,
    plot_health_scores,
    plot_topic_polarization,
    plot_virality_by_category,
    compute_overall_health_score,
)

st.set_page_config(page_title="Social Media Platform Analysis Dashboard", layout="wide")
st.title("🧪 Social Media Platform Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your social_data.csv", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
        df['comments'] = pd.to_numeric(df['comments'], errors='coerce')
        df['shares'] = pd.to_numeric(df['shares'], errors='coerce')

        echo_res = analyze_echo_chambers(df)
        pol_res = analyze_polarization(df)
        alg_bias = analyze_algorithmic_bias(df)
        misinfo = analyze_misinformation(df)
        net_res = analyze_network_structure(df)
        bias_scores = [
            echo_res['diversity_stats']['mean'],
            1 - min(pol_res['polarization_score'] / 1.0, 1.0),
            min(alg_bias['bias_score'], 2.0) / 2.0,
            min(misinfo['amplification_ratio'], 3.0) / 3.0 if misinfo['amplification_ratio'] is not None else 0
        ]
        overall = compute_overall_health_score({
            'echo_chambers': echo_res,
            'polarization': pol_res,
            'algorithmic_bias': alg_bias,
            'misinformation': misinfo,
        })

        st.header("Summary Metrics")
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Echo Chamber Strength", f"{echo_res['diversity_stats']['mean']:.3f}")
        metrics_cols[1].metric("Polarization Level", f"{pol_res['polarization_score']:.3f}")
        metrics_cols[2].metric("Algorithmic Bias", f"{alg_bias['bias_score']:.3f}")
        metrics_cols[3].metric("Misinformation Spread", f"{misinfo['amplification_ratio'] if misinfo['amplification_ratio'] else 0:.3f}")

        st.subheader("Overall Platform Health Score")
        st.info(f"{overall:.1f}/100")
        if overall > 80:
            st.success("✅ EXCELLENT: Platform shows healthy discourse patterns")
        elif overall > 60:
            st.warning("⚠️ MODERATE: Some concerning patterns detected")
        else:
            st.error("❌ POOR: Significant platform health issues detected")

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Polarization", "Algorithmic Bias", "Misinformation", "Echo Chambers", "Categories", "Scores"
        ])

        with tab1:
            st.subheader("Sentiment Distribution")
            st.pyplot(plot_sentiment_distribution(df))
            st.subheader("Most Polarized Topics")
            st.pyplot(plot_topic_polarization(pol_res['topic_polarization']))

        with tab2:
            st.subheader("Engagement by Category")
            st.pyplot(plot_engagement_by_category(df))
            st.subheader("Virality by Category")
            st.pyplot(plot_virality_by_category(alg_bias['virality_data']))

        with tab3:
            st.subheader("Temporal Spread (Misinformation vs Safe)")
            st.pyplot(plot_temporal_content_spread(df))

        with tab4:
            st.subheader("User Content Diversity")
            st.pyplot(plot_user_content_diversity(df))
            if net_res:
                st.metric("Network Density", f"{net_res['density']:.3f}")
                st.metric("Avg Clustering", f"{net_res['avg_clustering']:.3f}")
                st.metric("Avg Degree", f"{net_res['avg_degree']:.1f}")
            else:
                st.warning("Network too small for analysis.")

        with tab5:
            st.subheader("Category Distribution")
            st.pyplot(plot_category_distribution(df))

        with tab6:
            st.subheader("Normalized Platform Health Scores")
            st.pyplot(plot_health_scores(bias_scores))
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.warning("Please upload your social_data.csv file to get started.")

