# Development Guide

Setup, contribution guidelines, and development workflow for TruthMindr-Agent.

---

## Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA 11+ for GPU support)
- Tesseract-OCR system library
- 8+ GB RAM (16 GB recommended)
- GPU recommended but not required

---

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/Grewal-Pam/truthminder-agent.git
cd truthminder-agent
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n truthminder python=3.9 -y
conda activate truthminder

# Or using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
- Download installer from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki/Downloads)

### 5. Set Environment Variables
```bash
export TRUTHMINDR_TASK="3way"
export TRUTHMINDR_ABSTAIN_T="0.45"
export NEWSAPI_KEY="your_api_key_here"
export TRANSFORMERS_OFFLINE=0        # Download models from HuggingFace
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
```

### 6. Verify Installation
```bash
python -c "
import torch
import transformers
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ GPU: {torch.cuda.is_available()}')
"
```

---

## Project Structure for Development

```
truthminder-agent/
├── agent/              # Orchestration logic
│   └── runner.py       # Main inference pipeline
├── models/             # Inference modules
│   ├── clip_infer.py
│   ├── vilt_infer.py
│   └── flava_infer.py
├── tools/              # Utility functions
│   ├── ocr.py
│   ├── consistency.py
│   ├── retrieval.py
│   └── nli.py
├── app/                # Streamlit interface
│   └── test_app.py
├── etl/                # Data pipeline
│   ├── ingest/
│   ├── transform/
│   ├── load/
│   └── pipeline.py
├── training/           # Model fine-tuning
├── evaluate/           # Model evaluation
├── tests/              # Unit tests
├── config.py           # Configuration
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

---

## Development Workflow

### 1. Running Individual Components

#### Test CLIP Model
```python
from models.clip_infer import run_clip

result = run_clip(
    "https://example.com/image.jpg",
    "News headline text",
    task="3way"
)
print(result)
```

#### Test OCR
```python
from tools.ocr import ocr

text = ocr("https://example.com/image.jpg")
print(text)
```

#### Test NLI
```python
from tools.consistency import build_consistency_features

features = build_consistency_features(
    "image_url",
    "title",
    "ocr_text"
)
print(features)
```

### 2. Testing Single Row
```python
from agent.runner import run_row

row = {
    "id": "test_123",
    "image_url": "https://...",
    "clean_title": "News headline",
    "upvote_ratio": 0.85,
    "score": 100,
    "num_comments": 10
}

result, trace = run_row(row)
print("Result:", result)
print("Trace:", trace)
```

### 3. Running Streamlit App
```bash
streamlit run app/test_app.py --server.port 8501
```

Open browser at http://localhost:8501

### 4. Running ETL Pipeline
```bash
# Ingest data from all sources
python -m etl.pipeline

# With custom parameters
python -c "from etl.pipeline import run; run(limit=100, country='gb')"
```

---

## Code Style & Standards

### Python Style Guide
Follow PEP 8:
```bash
# Format code
black .

# Check style
flake8 .

# Type checking (optional)
mypy agent/ models/ tools/
```

### Documentation
- Use docstrings for all functions
- Follow Google docstring format:

```python
def run_clip(image_url: str, title: str, task: str = "3way") -> dict:
    """
    Run CLIP inference on image-text pair.
    
    Args:
        image_url: URL or local path to image
        title: Text caption for the image
        task: Classification task ("2way" or "3way")
    
    Returns:
        Dictionary with probabilities for each label
        
    Example:
        >>> result = run_clip("image.jpg", "News title")
        >>> print(result)
        {'Real': 0.72, 'Satire': 0.15, 'Fake': 0.13}
    """
```

### Commit Messages
Follow conventional commits:
```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
refactor: Code refactoring
test: Add/update tests
chore: Maintenance tasks

Example:
feat: Add RAG evidence retrieval layer
- Integrate FAISS for semantic search
- Add LangChain RetrievalQA
- Update arbiter to use evidence
```

---

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py::test_clip_inference

# With coverage
pytest --cov=models tests/
```

### Integration Tests
```bash
# Test full pipeline
python agent/runner.py --test-mode --test-file data/test_batch_1.tsv
```

### Manual Testing Checklist
- [ ] CLIP inference works
- [ ] ViLT inference works
- [ ] FLAVA inference works
- [ ] OCR extraction works
- [ ] NLI consistency check works
- [ ] Retrieval works
- [ ] Streamlit app launches
- [ ] Batch processing works
- [ ] Output files generated correctly

---

## Adding New Features

### Example: Adding a New Model

1. **Create inference module**
```python
# models/new_model_infer.py
def run_new_model(image_url: str, title: str, task: str = "3way") -> dict:
    """Run new model inference."""
    # Implementation
    pass
```

2. **Import in agent/runner.py**
```python
from models.new_model_infer import run_new_model
```

3. **Add to load_models()**
```python
def load_models():
    return {
        "clip_fn": run_clip,
        "vilt_fn": run_vilt,
        "flava_fn": run_flava,
        "new_fn": run_new_model,  # Add here
    }
```

4. **Update run_row() if needed**
```python
def run_row(row: dict, models=None):
    # ...
    new_pred = models["new_fn"](...)
    # ...
```

5. **Update Streamlit app** if displaying predictions
6. **Add tests** for new model
7. **Update documentation**

---

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from agent.runner import run_row
# Now you'll see detailed logs
```

### Profile Inference Speed
```python
import time
from agent.runner import run_row

row = {...}
start = time.time()
result, trace = run_row(row)
elapsed = time.time() - start
print(f"Inference took {elapsed:.2f} seconds")
```

### Check Memory Usage
```bash
# Monitor while running
watch -n 1 nvidia-smi  # GPU
watch -n 1 free -h     # CPU RAM
```

### Debug with pdb
```python
import pdb

def my_function():
    pdb.set_trace()  # Execution pauses here
    # Now inspect variables, step through code
```

---

## Common Development Tasks

### Update Dependencies
```bash
# Check for updates
pip list --outdated

# Update specific package
pip install --upgrade torch transformers

# Update all
pip install -U -r requirements.txt

# Save updated versions
pip freeze > requirements.txt
```

### Add New Dependency
```bash
# Install
pip install new_package

# Add to requirements.txt
echo "new_package==1.0.0" >> requirements.txt

# Commit changes
git add requirements.txt
git commit -m "chore: Add new_package dependency"
```

### Clear Cache
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Clear FAISS index
rm data/faiss_index

# Clear logs
rm -rf logs/*
```

---

## Performance Optimization

### Batch Processing
```python
# Process multiple rows efficiently
from agent.runner import run_file

run_file("data/test_batch_1.tsv", out_dir="outputs")
# Uses model caching for 20-30% speedup
```

### GPU Acceleration
```python
# Ensure GPU is used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Set memory growth to avoid OOM
torch.cuda.set_per_process_memory_fraction(0.7)
```

### Model Quantization (Future)
```python
# For edge deployment, quantize models
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(...)
# Apply quantization via torch.quantization
```

---

## Troubleshooting

### "Module not found" error
```bash
# Ensure you're in correct directory
cd truthminder-agent

# Reinstall in development mode
pip install -e .
```

### GPU out of memory
```python
# Reduce batch size
BATCH_SIZE = 8  # Instead of 32

# Or use CPU
import torch
torch.device("cpu")
```

### Tesseract not found
```bash
# Install again
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# Or set path manually
import pytesseract
pytesseract.pytesseract.pytesseract_cmd = '/usr/bin/tesseract'
```

### Models won't download
```bash
# Check HuggingFace token
huggingface-cli login

# Or use offline mode
export TRANSFORMERS_OFFLINE=1
```

---

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Git Workflow Guide](https://guides.github.com/introduction/flow/)
- [Python PEP 8](https://www.python.org/dev/peps/pep-0008/)

---

## Getting Help

1. Check documentation: `README.md`, `ARCHITECTURE.md`
2. Search issues: GitHub Issues
3. Review code comments and docstrings
4. Ask on discussions: GitHub Discussions
5. Contact maintainers: GitHub profile

---

**Happy coding! 🚀**
