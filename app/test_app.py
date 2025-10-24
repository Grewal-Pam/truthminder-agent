# truthmindr_agent/app/test_app.py
import os
import streamlit as st
import pandas as pd
from PIL import Image

# Environment fixes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.set_page_config(page_title="TruthMindr", layout="wide")
st.title("📰 TruthMindr: Multimodal Disinformation Detector")
st.markdown(
    "🚀 *Please wait a moment while models initialize (first run may take 1–2 mins)*"
)


# -------------------------------------------------------------------
# Cache heavy imports (run_row, rephrase_result)
# -------------------------------------------------------------------
@st.cache_resource
def load_runner():
    from agent.runner import run_row

    return run_row


@st.cache_resource
def load_rephraser():
    from tools.rephraser import rephrase_result

    return rephrase_result


run_row = load_runner()
rephrase_result = load_rephraser()

# Initialize session state to prevent unnecessary re-runs
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None


def render_rag_section(rag_text: str):
    """Pretty RAG section with basic cleanup & fallback."""
    if not rag_text or not rag_text.strip():
        st.info("📚 No relevant evidence found in the current corpus.")
        return
    bland = ["i don't know", "i cannot", "not a doctor", "does not relate"]
    if any(b in rag_text.lower() for b in bland):
        st.info("📚 No strong matches in the corpus for this claim. (Consider enriching the knowledge base.)")
        return
    with st.expander("📚 Evidence from Knowledge Corpus (RAG)", expanded=False):
        st.markdown(rag_text)

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📂 Explore Saved Posts", "🧪 Try Your Own Post"])

# -------------------------------------------------------------------
# TAB 1: Explore pre-annotated enriched.csv
# -------------------------------------------------------------------
with tab1:
    st.subheader("Explore Pre-Annotated Posts")

    try:
        df = pd.read_csv("outputs/enriched.csv")
        selected_id = st.selectbox("Choose a Post ID", df["post_id"].unique())

        if selected_id:
            row = df[df["post_id"] == selected_id].iloc[0].to_dict()

            col1, col2 = st.columns([1, 2])
            with col1:
                if row.get("image_url") and str(row["image_url"]).startswith("http"):
                    st.image(
                        row["image_url"],
                        caption=row["clean_title"],
                        use_container_width=True,
                    )

            with col2:
                st.write(f"### {row['clean_title']}")
                st.markdown(f"**Final Label:** `{row['final_label']}`")
                st.markdown(f"**Confidence:** {row['final_confidence']:.2f}")
                st.caption(
                    f"Consistency Score: {row['consistency_score']:.2f}, NLI: {row['nli_label']}"
                )

            explanation = rephrase_result(row)
            st.info(explanation)
            # RAG for Tab 1 (row loaded from enriched.csv)
            render_rag_section(row.get("retrieved_text", ""))


    except FileNotFoundError:
        st.error("⚠️ No enriched.csv found in outputs/. Run the agent first.")

# -------------------------------------------------------------------
# TAB 2: Try a new post
# -------------------------------------------------------------------
with tab2:
    st.subheader("Test a New News Post")

    clean_title = st.text_input("News Title", "")
    task = st.selectbox("Classification Task", ["3way", "2way"])

    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("...or paste an Image URL")

    st.markdown("### Optional Metadata")
    num_comments = st.number_input("Number of Comments", value=0)
    score = st.number_input("Score", value=0)
    upvote_ratio = st.number_input("Upvote Ratio", value=0.0, step=0.01)

    # 🔍 Analyze button — only run models when pressed
    if st.button("🔍 Analyze"):
        if not clean_title or (not uploaded_img and not image_url):
            st.error("Please provide a Title and either upload an Image or paste URL.")
        else:
            st.session_state.analysis_done = True

            if uploaded_img:
                img = Image.open(uploaded_img).convert("RGB")
                local_path = "temp_upload.png"
                img.save(local_path)
                image_input = local_path
            else:
                image_input = image_url

            row = {
                "id": "user_post",
                "clean_title": clean_title,
                "image_url": image_input,
                "num_comments": num_comments,
                "score": score,
                "upvote_ratio": upvote_ratio,
                "2_way_label": 1,
                "3_way_label": 0,
            }

            with st.spinner("Running CLIP, ViLT, FLAVA + Arbiter..."):
                result, trace = run_row(row)

            # Cache results in session_state
            st.session_state.last_result = result
            st.session_state.last_trace = trace

    # 💾 Display results only if analysis is complete
    if st.session_state.analysis_done and st.session_state.last_result:
        result = st.session_state.last_result
        trace = st.session_state.last_trace

        col1, col2 = st.columns([1, 2])
        with col1:
            if uploaded_img:
                st.image(
                    Image.open("temp_upload.png"),
                    caption="Uploaded Image",
                    use_container_width=True,
                )
            elif image_url:
                st.image(image_url, caption="From URL", use_container_width=True)

        with col2:
            st.write(f"### {clean_title}")
            st.markdown(f"**Final Label:** `{result['final_label']}`")
            st.markdown(f"**Confidence:** {result['final_confidence']:.2f}")
            st.caption(
                f"Consistency Score: {result['consistency_score']:.2f}, NLI: {result['nli_label']}"
            )

        explanation = rephrase_result(result)
        st.success("✅ Analysis Complete")
        st.info(explanation)

        # RAG for Tab 2 (fresh run result)
        render_rag_section(result.get("retrieved_text", ""))

        # (Optional) If you want to fall back to the trace step:
        if (not result.get("retrieved_text")) and trace and "steps" in trace:
            ret = [s for s in trace["steps"] if s.get("stage")=="retrieval"]
            if ret:
                render_rag_section(ret[-1].get("retrieved_evidence",""))


        with st.expander("🔬 Technical Trace"):
            st.json(trace)

        # Add model comparison
        if "steps" in trace and len(trace["steps"]) > 0:
            votes = pd.DataFrame(
                [
                    {
                        "Model": "CLIP",
                        "Real": trace["steps"][0]["clip"]["Real"],
                        "Fake": trace["steps"][0]["clip"]["Fake"],
                    },
                    {
                        "Model": "ViLT",
                        "Real": trace["steps"][0]["vilt"]["Real"],
                        "Fake": trace["steps"][0]["vilt"]["Fake"],
                    },
                    {
                        "Model": "FLAVA",
                        "Real": trace["steps"][0]["flava"]["Real"],
                        "Fake": trace["steps"][0]["flava"]["Fake"],
                    },
                ]
            )
            st.bar_chart(votes.set_index("Model"))
