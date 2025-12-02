# Models Module

Inference modules for three pretrained vision-language transformer models: CLIP, ViLT, and FLAVA.

## Overview

Each model implements:
1. **Model Loading** - Load pretrained weights and setup for inference
2. **Data Preprocessing** - Handle images and text tokenization
3. **Inference** - Generate predictions for disinformation classification
4. **Output Format** - Return probabilities for Real/Satire/Fake labels

---

## CLIP Model

**File:** `clip_infer.py` / `clip_model.py`

### Overview
CLIP (Contrastive Language-Image Pre-training) is an open-source vision-language model that learns visual and textual representations jointly.

### Key Functions
```python
run_clip(image_url: str, title: str, task: str = "3way", metadata: dict = None) -> dict
```

### Output Format
```python
{
    "Real": 0.72,
    "Satire/Mixed": 0.15,
    "Fake": 0.13
}
```

### Configuration
- Model: `openai/clip-vit-base-patch32`
- Image size: 224×224
- Batch processing: Supported
- Device: Auto-detects (GPU/CPU)

---

## ViLT Model

**File:** `vilt_infer.py` / `vilt_model.py`

### Overview
ViLT (Vision-and-Language Transformer) uses a unified transformer architecture for efficient multimodal processing.

### Key Functions
```python
run_vilt(image_url: str, title: str, task: str = "3way", metadata: dict = None) -> dict
```

### Output Format
```python
{
    "Real": 0.68,
    "Satire/Mixed": 0.22,
    "Fake": 0.10
}
```

### Configuration
- Model: `dandelin/vilt-b32-mlm` (vision-language)
- Image size: 384×384
- Fastest inference among the three
- Best for real-time applications

---

## FLAVA Model

**File:** `flava_infer.py` / `flava_model.py`

### Overview
FLAVA (Foundational Language And Vision Alignment) is Facebook's multimodal foundation model with strong metadata integration.

### Key Functions
```python
run_flava(image_url: str, title: str, task: str = "3way", metadata: dict = None) -> dict
```

### Output Format
```python
{
    "Real": 0.70,
    "Satire/Mixed": 0.18,
    "Fake": 0.12
}
```

### Configuration
- Model: `facebook/flava-full`
- Image size: 256×256
- Best metadata sensitivity
- Strongest across all modalities

---

## Usage Example

```python
from models.clip_infer import run_clip
from models.vilt_infer import run_vilt
from models.flava_infer import run_flava

image_url = "https://example.com/image.jpg"
title = "Breaking: Important News"
task = "3way"  # or "2way"

# Get predictions from all models
clip_pred = run_clip(image_url, title, task)
vilt_pred = run_vilt(image_url, title, task)
flava_pred = run_flava(image_url, title, task)

print(f"CLIP: {clip_pred}")
print(f"ViLT: {vilt_pred}")
print(f"FLAVA: {flava_pred}")
```

---

## Model Comparison

| Aspect | CLIP | ViLT | FLAVA |
|--------|------|------|-------|
| Architecture | Vision-Language | Unified Transformer | Multimodal Foundation |
| Inference Speed | Medium | **Fast** | Medium |
| Accuracy (2-way) | **92.4%** | 88.7% | 90.1% |
| Accuracy (3-way) | **80.2%** | 75.2% | 78.6% |
| Metadata Handling | Limited | Moderate | **Best** |
| Recommended Use | Visual reasoning | Real-time inference | Balanced performance |

---

## Fine-tuning

Each model directory contains training scripts:
- `clip_model.py` - Training utilities for CLIP
- `vilt_model.py` - Training utilities for ViLT
- `flava_model.py` - Training utilities for FLAVA

See `training/` directory for full training pipeline.

---

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size or use gradient checkpointing
BATCH_SIZE = 16
```

### Model Download Fails
```bash
export HF_HOME="/path/to/cache"
```

### Slow Inference
Use ViLT for fastest inference, or enable GPU acceleration.

---

## References

- [CLIP Paper](https://arxiv.org/abs/2103.14030)
- [ViLT Paper](https://arxiv.org/abs/2102.03334)
- [FLAVA Paper](https://arxiv.org/abs/2112.04482)
