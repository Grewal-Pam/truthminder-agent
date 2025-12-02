"""
TruthMindr-Agent Configuration Module

Central configuration for all constants, API keys, model paths, and settings.
"""

import os
import torch

# ===== Task Configuration =====
TASK = os.environ.get("TRUTHMINDR_TASK", "3way")  # "2way" or "3way"
ABSTAIN_T = float(os.environ.get("TRUTHMINDR_ABSTAIN_T", "0.45"))

# ===== API Keys =====
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
PRAW_CLIENT_ID = os.environ.get("PRAW_CLIENT_ID", "")
PRAW_CLIENT_SECRET = os.environ.get("PRAW_CLIENT_SECRET", "")

# ===== Device Configuration =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

# ===== Model Configuration =====
MODEL_CHECKPOINT_DIR = "models/checkpoints/"
CLIP_MODEL = "openai/clip-vit-base-patch32"
VILT_MODEL = "dandelin/vilt-b32-mlm"
FLAVA_MODEL = "facebook/flava-full"

# ===== Data Paths =====
DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"
RESULTS_DIR = "results/"
LOGS_DIR = "logs/"
MODELS_DIR = "models/"

# ===== Model Inference Settings =====
BATCH_SIZE = 32
IMAGE_SIZE = 224
MAX_SEQ_LENGTH = 77

# ===== OCR Configuration =====
OCR_CONFIDENCE_THRESHOLD = 0.5
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "tesseract")

# ===== NLI Configuration =====
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_CONFIDENCE_THRESHOLD = 0.5

# ===== Retrieval Configuration =====
RETRIEVAL_TOP_K = 5
FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ===== Streamlit Settings =====
STREAMLIT_PORT = 8501
STREAMLIT_THEME = "dark"

# ===== Training Configuration =====
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01

# ===== Logging Configuration =====
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ===== Metadata Fields =====
METADATA_FIELDS = [
    "upvote_ratio",
    "score",
    "num_comments",
    "timestamp",
    "source"
]

# ===== Classification Labels =====
LABELS_2WAY = ["Real", "Fake"]
LABELS_3WAY = ["Real", "Satire/Mixed", "Fake"]


def get_config(key, default=None):
    """Get configuration value by key name."""
    return globals().get(key, default)


def validate_config():
    """Validate critical configuration settings."""
    issues = []
    
    if not NEWSAPI_KEY:
        issues.append("NEWSAPI_KEY not set - NewsAPI source will fail")
    
    if not os.path.exists(DATA_DIR):
        issues.append(f"DATA_DIR does not exist: {DATA_DIR}")
    
    if USE_GPU:
        print("✅ GPU available - inference will be accelerated")
    else:
        print("⚠️  GPU not available - using CPU (slower inference)")
    
    if issues:
        print("⚠️  Configuration warnings:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0


if __name__ == "__main__":
    validate_config()
