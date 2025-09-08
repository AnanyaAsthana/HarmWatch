import streamlit as st
import pandas as pd

st.title("Insta-Research Dashboard — Demo")
st.write("Replace this with your real dashboard components.")

# Example: load a CSV if hosted or included
if st.button("Load demo CSV"):
    df = pd.DataFrame({
        "post_id":["p1","p2"],
        "user_id":["u1","u2"],
        "timestamp":["2025-09-01","2025-09-02"],
        "post_text":["hello","world"],
        "hashtags":["#a","#b"],
        "category":["Safe","Misinformation"],
        "sentiment":[0.1,-0.6],
    })
    st.dataframe(df)
