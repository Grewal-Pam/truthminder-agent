import streamlit as st
import pandas as pd
import os
from tools.rephraser import rephrase_result

DATA_PATH = "outputs/enriched.csv"

st.set_page_config(page_title="TruthMindr Demo", page_icon="🕵️", layout="wide")

st.title("📰 TruthMindr: News Reality Checker")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

@st.cache_data(show_spinner=False)
def get_explanation(row_dict):
    return rephrase_result(row_dict)

df = load_data()
if df is None:
    st.error("No enriched.csv found. Please run agent/runner.py first.")
else:
    # Sidebar navigation
    post_ids = df["post_id"].tolist()
    choice = st.sidebar.selectbox("Choose a post ID", post_ids)

    row = df[df["post_id"] == choice].iloc[0].to_dict()

    st.subheader(row["clean_title"])

    if row.get("image_url") and str(row["image_url"]).startswith("http"):
        st.image(row["image_url"], width=400)

    verdict_color = {"Real": "🟢", "Fake": "🔴", "SatireMixed": "🟡"}.get(row["final_label"], "⚪")
    st.markdown(f"### Verdict: {verdict_color} **{row['final_label']}** (Confidence: {row['final_confidence']:.2f})")

    with st.spinner("Explaining..."):
        explanation = get_explanation(row)
    st.write(f"💡 {explanation}")

    if st.checkbox("Show technical details"):
        st.json({
            "Consistency score": row["consistency_score"],
            "Clip cosine": row["clip_cos"],
            "NLI label": row["nli_label"],
            "OCR text": row["ocr_text"],
            "Metadata": {
                "upvote_ratio": row["upvote_ratio"],
                "score": row["score"],
                "num_comments": row["num_comments"]
            }
        })
