# Architecture & Design Document

Comprehensive technical architecture of TruthMindr-Agent system.

---

## System Overview

TruthMindr-Agent is a modular, multi-layered system for multimodal disinformation detection:

```
┌─────────────────────────────────────────────────────────┐
│              User Interface (Streamlit)                  │
│         - Single post analysis                           │
│         - Batch processing                               │
│         - Interactive visualizations                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│          Agent Orchestration Layer                       │
│  (agent/runner.py - run_row, run_file)                  │
│  Coordinates all reasoning stages                        │
└──┬───────────┬──────────────┬──────────────┬────────────┘
   │           │              │              │
   │           │              │              │
┌──▼─┐   ┌─────▼──┐  ┌──────▼───┐  ┌───────▼──┐
│ P  │   │    E   │  │     C    │  │    A     │
│ e  │   │        │  │          │  │          │
│ r  │   │Evidence │  │onsist.  │  │rbiter    │
│ c. │   │ (OCR)  │  │(NLI+Sim) │  │(Fusion)  │
└─┬──┘   └────┬───┘  └────┬─────┘  └───────┬──┘
  │           │           │                │
  │ ┌─────────▼──────┐    │                │
  ├─►│ CLIP          │    │                │
  │ │ ViLT          │    │                │
  │ │ FLAVA         │    │                │
  │ └────────────────┘    │                │
  │                       │                │
  │         ┌─────────────▼────────────┐   │
  │         │  Consistency Features    │   │
  │         │  - NLI label            │   │
  │         │  - CLIP cosine sim      │   │
  │         └──────────┬───────────────┘   │
  │                    │                   │
  │         ┌──────────▼────────────┐  ┌───▼─────────┐
  │         │  Retrieval (RAG)      │  │ Final       │
  │         │  - FAISS search       │  │ Decision    │
  │         │  - LLM context        │  │ + Confidence│
  │         └───────────────────────┘  └─────────────┘
  │
  └──────────────────────────────┐
                                  │
                          ┌───────▼────────┐
                          │  Output        │
                          │- enriched.csv  │
                          │- traces/*.json │
                          └────────────────┘
```

---

## Core Components

### 1. **Perception Layer** (Models)

Three pretrained vision-language models process image-text pairs:

#### CLIP (Contrastive Language-Image Pre-training)
- **Model:** `openai/clip-vit-base-patch32`
- **Input:** Image URL + Text caption
- **Process:**
  1. Download image from URL
  2. Resize to 224×224
  3. Get image embeddings from ViT-Base
  4. Get text embeddings from BERT-like tokenizer
  5. Compute similarity → classify
- **Output:** Probabilities for [Real, Satire/Fake]
- **Strengths:** Best visual reasoning, highest accuracy

#### ViLT (Vision-and-Language Transformer)
- **Model:** `dandelin/vilt-b32-mlm`
- **Process:**
  1. Unified transformer processes image patches + text tokens
  2. Single forward pass for image & text
  3. Classification head → labels
- **Output:** Probabilities for [Real, Satire, Fake]
- **Strengths:** Fastest inference, lightweight

#### FLAVA (Foundational Language And Vision Alignment)
- **Model:** `facebook/flava-full`
- **Process:**
  1. Separate encoders for image & text
  2. Fusion layer combines modalities
  3. Classification head outputs
- **Output:** Probabilities for [Real, Satire, Fake]
- **Strengths:** Best metadata integration, most balanced

---

### 2. **Evidence Layer** (OCR)

Extracts text from image using Tesseract OCR:

```python
Input: Image URL
  │
  ├─► Download image
  ├─► Preprocess (resize, normalize)
  ├─► Run Tesseract OCR
  ├─► Post-process (clean, normalize)
  │
Output: Extracted text string (or "")
```

**Key Functions:**
- `ocr(path_or_url)` → str
- Handles both local files and URLs
- Returns empty string if OCR fails

---

### 3. **Consistency Layer** (NLI + Similarity)

Assesses alignment between image, text, and caption:

#### A. CLIP Image-Text Cosine Similarity
```python
Input: Image URL + Caption
  │
  ├─► Get image embedding via CLIP
  ├─► Get caption embedding via CLIP
  ├─► Compute cosine similarity
  │
Output: Float [0, 1] (higher = more aligned)
```

#### B. Natural Language Inference (NLI)
```python
Input: Caption + OCR text
  │
  ├─► "Does caption follow from OCR text?"
  ├─► Cross-encoder (DeBERTa) evaluates
  ├─► Returns: ENTAILMENT, NEUTRAL, CONTRADICTION
  │
Output: Label + confidence score
```

#### C. Combined Consistency Score
```python
consistency_score = 0.6 * clip_similarity + 0.4 * nli_confidence
```

---

### 4. **Retrieval Layer** (RAG)

Retrieves supporting or contradicting evidence:

```python
Input: Post title + OCR text
  │
  ├─► Concatenate into query
  ├─► Encode query to embeddings
  ├─► Search FAISS index for similar documents
  ├─► Retrieve top-K documents
  ├─► Feed to LLM for summarization
  │
Output: Evidence text summary
```

**Components:**
- **FAISS Index:** Fast similarity search
- **Embedding Model:** Sentence-transformers
- **LLM:** LangChain RetrievalQA

---

### 5. **Arbiter Layer** (Decision Fusion)

Combines all reasoning into final decision:

```python
Input: 
  - clip_probs: [Real, Satire, Fake]
  - vilt_probs: [Real, Satire, Fake]
  - flava_probs: [Real, Satire, Fake]
  - consistency_score: float
  - metadata: {upvotes, score, comments}
  
Process:
  1. Average model probabilities (ensemble)
  2. Weight by consistency score
  3. Incorporate metadata signals
  4. Find argmax → final label
  5. Compute confidence

Output:
  - label: "Real" | "Satire/Mixed" | "Fake"
  - confidence: float [0, 1]
  - abstain: bool (if confidence < threshold)
```

**Abstention Mechanism:**
- If max confidence < ABSTAIN_THRESHOLD (0.45)
- Return "ABSTAIN" instead of risky prediction
- Allows for uncertain cases

---

## Data Flow

### Single Row Processing

```
input_row = {
    "id": "post_123",
    "image_url": "http://...",
    "clean_title": "News headline",
    "upvote_ratio": 0.85,
    "score": 123,
    "num_comments": 45
}
    │
    ├──► load_models() [cached after first load]
    │
    ├──► Perception Layer
    │    ├─ run_clip() → {Real: 0.72, ...}
    │    ├─ run_vilt() → {Real: 0.68, ...}
    │    └─ run_flava() → {Real: 0.70, ...}
    │
    ├──► Evidence Layer
    │    └─ ocr(image_url) → "extracted text"
    │
    ├──► Consistency Layer
    │    ├─ clip_image_text_cosine() → 0.75
    │    └─ build_consistency_features() → {nli_label, score}
    │
    ├──► Retrieval Layer
    │    └─ retrieve_evidence(query) → "supporting text"
    │
    ├──► Arbiter Layer
    │    └─ arbiter(clip_p, vilt_p, flava_p, cons, meta) 
    │        → ("Real", 0.86)
    │
    └──► Output
         ├─ result = enriched row + predictions
         └─ trace = full reasoning JSON
```

### Batch Processing

```
input_file.tsv (1000 rows)
    │
    ├─► load_models() once
    ├─► For each row:
    │    └─ run_row(row, models)
    │        └─ Generate result + trace
    │
    └──► Output
         ├─ enriched.csv (all predictions)
         └─ traces/ folder (per-post JSON)
```

---

## Data Structures

### Input Row Schema
```python
{
    "id": str,                    # Unique post ID
    "image_url": str,             # URL to image
    "clean_title": str,           # Text/caption
    "upvote_ratio": float,        # Engagement metric [-1, 1]
    "score": float,               # Post score
    "num_comments": int,          # Comment count
    "source": str,                # reddit|twitter|newsapi
    [other metadata fields...]
}
```

### Output Result Schema
```python
{
    "post_id": str,
    "final_label": str,           # Real | Satire | Fake | ABSTAIN
    "final_confidence": float,    # [0, 1]
    "clip_pred": dict,            # Model probabilities
    "vilt_pred": dict,
    "flava_pred": dict,
    "nli_label": str,             # ENTAILMENT | NEUTRAL | CONTRADICTION
    "consistency_score": float,
    "clip_cos": float,            # Image-text similarity
    "ocr_text": str,              # Extracted text
    "retrieved_text": str,        # RAG evidence
    [other enriched fields...]
}
```

### Trace JSON Schema
```json
{
    "post_id": "...",
    "steps": [
        {
            "stage": "perception",
            "clip": {...},
            "vilt": {...},
            "flava": {...}
        },
        {
            "stage": "evidence",
            "ocr_text": "...",
            "no_ocr": false
        },
        {
            "stage": "consistency",
            "nli_label": "...",
            "clip_cos": 0.75,
            "consistency_score": 0.82
        },
        {
            "stage": "retrieval",
            "retrieved_evidence": "..."
        }
    ],
    "final": {
        "post_id": "...",
        "final_label": "Real",
        "final_confidence": 0.86,
        ...
    }
}
```

---

## Deployment Architectures

### Local Development
```
├─ Python environment (conda/venv)
├─ Models cached locally (~10 GB)
├─ Streamlit dev server
└─ SQLite database (truthmindr.db)
```

### Production (Docker)
```
Docker Container
├─ Python 3.9 + dependencies
├─ Preloaded models
├─ FastAPI server
└─ PostgreSQL backend
```

### Distributed (Kubernetes)
```
Kubernetes Cluster
├─ Model serving pods (multiple replicas)
├─ Inference load balancer
├─ Cache layer (Redis)
└─ Persistent volume for models
```

---

## Performance Characteristics

### Inference Time (per post)
- CLIP: ~150 ms (GPU) / ~1s (CPU)
- ViLT: ~80 ms (GPU) / ~600 ms (CPU)
- FLAVA: ~120 ms (GPU) / ~800 ms (CPU)
- OCR: ~300-500 ms (depends on image quality)
- NLI: ~150 ms
- Retrieval: ~50 ms (FAISS search) + LLM generation

**Total**: ~1.5-2.5 seconds end-to-end on GPU

### Memory Requirements
- Models: ~8 GB (all three models loaded)
- Batch processing (32): ~4 GB additional
- FAISS index: Size depends on corpus

### Scalability
- **Single machine:** ~100 posts/minute on GPU
- **Kubernetes:** Horizontal scaling via pod replicas
- **Batch processing:** ~1000 posts/minute with optimal batching

---

## References

- Architecture inspired by: [TruthMindr Thesis](https://dl.acm.org/doi/10.1145/3746275.3762205)
- Model papers: CLIP, ViLT, FLAVA
- RAG framework: LangChain
- Vector search: FAISS
