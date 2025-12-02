# Streamlit Application

Interactive web-based interface for TruthMindr-Agent disinformation detection system.

## Files

- `test_app.py` - Main Streamlit application (215 lines)
- `streamlit_app.py` - Alternative app entry point (62 lines)

## Quick Start

```bash
# Activate environment
conda activate truthminder

# Run the app
streamlit run app/test_app.py --server.port 8501
```

**Access at:** http://localhost:8501

---

## Features

### 1. Single Post Analysis
- Upload image or provide image URL
- Enter post title/caption
- Get predictions from CLIP, ViLT, FLAVA models
- View reasoning traces (Perception → Evidence → Consistency → Arbiter)

### 2. Model Predictions Display
- Confidence scores for each model
- Visual confidence indicators (bars/gauges)
- Model-specific explanations

### 3. Reasoning Layer Transparency
- **Perception:** Raw model predictions
- **Evidence:** OCR text extraction
- **Consistency:** NLI label + CLIP similarity score
- **Arbiter:** Final decision with confidence

### 4. Evidence Retrieval
- Retrieved supporting/contradicting evidence from knowledge base
- Context for decision-making
- Source citations (if available)

### 5. Metadata Integration
- Post engagement metrics (upvotes, score, comments)
- Timestamp information
- Source (Reddit, Twitter, etc.)

---

## User Interface Components

### Input Section
```python
# Image input
image_source = st.radio("Image source:", ["URL", "Upload"])
if image_source == "URL":
    image_url = st.text_input("Enter image URL:")
else:
    uploaded_file = st.file_uploader("Upload image:")

# Text input
title = st.text_area("Post title/caption:")

# Metadata (optional)
st.expander("Advanced - Add Metadata")
    upvote_ratio = st.slider("Upvote ratio:", -1.0, 1.0)
    score = st.number_input("Score:")
```

### Output Display
```python
# Results in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("CLIP", f"{clip_conf:.1%}")
with col2:
    st.metric("ViLT", f"{vilt_conf:.1%}")
with col3:
    st.metric("FLAVA", f"{flava_conf:.1%}")

# Reasoning trace
with st.expander("View Reasoning Trace"):
    st.json(trace)
```

---

## Configuration

### Environment Variables
```bash
export TRANSFORMERS_OFFLINE=0        # Download models from HuggingFace
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
```

### Streamlit Settings (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
port = 8501
maxUploadSize = 50
```

---

## Dependencies

All required packages listed in `requirements.txt`:

```
streamlit==1.50.0
torch==2.0.1
transformers>=4.20.0
Pillow>=9.0
pandas>=1.5.0
pytesseract>=0.3.10
```

---

## Performance Tips

1. **Model Caching**
   - Models are cached after first load
   - Subsequent predictions faster

2. **Batch Processing**
   - Disable for single posts (faster response)
   - Enable for bulk analysis

3. **GPU Acceleration**
   - Requires CUDA 11.0+
   - Automatic detection in agent/runner.py

---

## Troubleshooting

### Models download fails
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Port already in use
```bash
streamlit run app/test_app.py --server.port 8502
```

### Memory error with large images
- Images automatically resized to 224×224 for CLIP
- Use `--logger.level=debug` for diagnostics

### Slow inference
- Ensure GPU is available: `torch.cuda.is_available()`
- Check `nvidia-smi` for GPU usage

---

## Deployment

### Local Development
```bash
streamlit run app/test_app.py
```

### Production (using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8501 app.test_app:app
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/test_app.py"]
```

---

## Architecture

```
User Input (Image + Title)
    ↓
Streamlit Session State
    ↓
Agent Runner (agent/runner.py)
    ├─ Load Models
    ├─ Run Perception (CLIP, ViLT, FLAVA)
    ├─ Run Evidence (OCR)
    ├─ Run Consistency (NLI + Similarity)
    ├─ Run Retrieval (RAG)
    └─ Run Arbiter (Decision Fusion)
    ↓
Display Results + Trace
```

---

## File Structure

```
app/
├── test_app.py          # Main Streamlit app
├── streamlit_app.py     # Alternative entry point
├── test_flava_predict.py # FLAVA testing script
├── test_app.py          # App testing utilities
└── assets/              # Images, CSS, static files
```

---

## Future Enhancements

- Real-time model streaming output
- Batch file upload (CSV/TSV)
- Model ensemble visualization
- Confidence calibration curves
- User feedback collection
- Export results to PDF

---

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-cloud/deploy-an-app)
