# Tools Module

Auxiliary reasoning tools for multimodal disinformation detection: OCR, NLI, retrieval, and explainability utilities.

## Overview

The tools module provides specialized functions for:
1. **OCR Extraction** - Extract text from images
2. **Consistency Checking** - Natural Language Inference (NLI) between title and OCR
3. **Text Similarity** - CLIP-based image-text cosine similarity
4. **Evidence Retrieval** - Retrieve supporting/contradicting evidence via RAG
5. **Explainability** - Generate explanations for model predictions

---

## OCR Module

**File:** `ocr.py`

### Function
```python
ocr(path_or_url: str) -> str
```

### Description
Extracts text from image using Tesseract OCR engine. Handles both local file paths and URLs.

### Input
- `path_or_url` (str): Local file path or HTTP(S) URL to image

### Output
- Text extracted from image (str)
- Returns empty string if OCR fails or no text found

### Example
```python
from tools.ocr import ocr

text = ocr("https://example.com/image.jpg")
print(text)  # "Extracted text from image"
```

### Dependencies
- `pytesseract` - Python wrapper for Tesseract OCR
- `Pillow` (PIL) - Image processing
- System: Tesseract binary installed (`apt install tesseract-ocr`)

---

## Consistency Module

**File:** `consistency.py`

### Functions

#### 1. CLIP Image-Text Cosine Similarity
```python
clip_image_text_cosine(path_or_url: str, caption: str) -> Optional[float]
```

Computes cosine similarity between image and caption using CLIP embeddings.

**Output:** Float between 0 and 1 (higher = more aligned)

#### 2. Build Consistency Features
```python
build_consistency_features(path_or_url: str, caption: str, ocr_text: str) -> dict
```

Combines NLI and CLIP similarity to assess consistency.

**Output:**
```python
{
    "nli_label": "ENTAILMENT",      # ENTAILMENT, NEUTRAL, CONTRADICTION
    "nli_score": 0.95,
    "clip_cos": 0.72,
    "consistency_score": 0.835      # Normalized combination
}
```

### NLI Engine
- Model: `cross-encoder/nli-deberta-v3-large`
- Labels: ENTAILMENT, NEUTRAL, CONTRADICTION
- Checks if title logically follows from OCR text + image

### Example
```python
from tools.consistency import build_consistency_features

features = build_consistency_features(
    "https://example.com/image.jpg",
    "News headline",
    "Text extracted from image"
)
print(f"Consistency score: {features['consistency_score']}")
```

---

## Retrieval Module

**File:** `retrieval.py`

### Function
```python
retrieve_evidence(query_text: str) -> str
```

Retrieves supporting or contradicting evidence from a knowledge base using RAG.

### Components
- **Vector Index:** FAISS (Facebook AI Similarity Search)
- **Embedding Model:** Sentence transformers
- **LLM:** LangChain with RetrievalQA

### Process
1. Convert query to embeddings
2. Search FAISS index for similar documents
3. Retrieve top-K relevant documents
4. Generate answer using LLM context

### Output
- Summarized evidence text (str)
- Returns empty string if no relevant evidence found

### Configuration
```python
RETRIEVAL_TOP_K = 5          # Number of documents to retrieve
FAISS_INDEX_PATH = "data/faiss_index"
```

### Example
```python
from tools.retrieval import retrieve_evidence

query = "COVID-19 vaccine safety"
evidence = retrieve_evidence(query)
print(evidence)  # Retrieved evidence about vaccine safety
```

### Index Building
```bash
# Build FAISS index from corpus
python tools/rag_build_index.py --corpus-path data/corpus/
```

---

## NLI Module

**File:** `nli.py`

Low-level Natural Language Inference utilities using cross-encoders.

### Functions
```python
def infer_nli(premise: str, hypothesis: str) -> dict
    # Returns: {"label": "ENTAILMENT", "scores": {...}}

def batch_nli(pairs: List[tuple]) -> List[dict]
    # Batch inference for multiple text pairs
```

---

## Explainability Module

**File:** `explain.py`

Generates human-readable explanations for model predictions.

### Functions
```python
def explain_prediction(prediction: dict, trace: dict) -> str
    # Returns: Natural language explanation of prediction

def generate_confidence_explanation(confidence: float) -> str
    # Returns: Interpretation of confidence level
```

### Example
```python
from tools.explain import explain_prediction

explanation = explain_prediction(
    {"label": "Fake", "confidence": 0.92},
    trace_data
)
print(explanation)
# "Model is 92% confident this is fake news due to..."
```

---

## Rephraser Module

**File:** `rephraser.py`

Text rephrasing utilities for data augmentation.

---

## Tool Integration Flow

```
Input Image + Title + Metadata
    ↓
[OCR] → Extract text from image
    ↓
[Consistency] → Compute NLI + CLIP similarity
    ↓
[Retrieval] → Find supporting/contradicting evidence
    ↓
[Output] → Combined features for Arbiter
```

---

## Dependencies

```bash
# Core tools
pytesseract          # OCR
transformers         # NLI models
sentence-transformers # Embeddings
faiss-cpu            # Vector search (or faiss-gpu)
langchain            # RAG orchestration
Pillow               # Image processing
```

---

## Troubleshooting

### OCR fails with "Tesseract not found"
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Set path if needed
import pytesseract
pytesseract.pytesseract.pytesseract_cmd = '/usr/bin/tesseract'
```

### NLI models slow on CPU
- Use GPU: `device = torch.device("cuda")`
- Use lightweight model: `microsoft/deberta-base`

### FAISS index not found
- Build index: `python tools/rag_build_index.py`
- Check `FAISS_INDEX_PATH` in config

---

## References

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract)
- [DeBERTa NLI](https://huggingface.co/cross-encoder/nli-deberta-v3-large)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://langchain.com/)
