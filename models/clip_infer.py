import os, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from models.clip_model import CLIPMultiTaskClassifier  # your class
from experiment_manager import prepare_dataloader      # your loader
from typing import Optional, Dict, Any


DEVICE = os.environ.get("TRUTHMINDR_DEVICE", "cpu")
METADATA_COLUMNS = ["num_comments", "score", "upvote_ratio"]
CHECKPOINT = "runs/CLIP/2025-04-13_07-20-33/models/clip_best_model_with_metadata.pth"
INPUT_DIM = 512

# Load once
_clip_proc = None
_clip_base = None
_clip_model = None

def _lazy_load():
    global _clip_proc, _clip_base, _clip_model
    if _clip_proc is None:
        _clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
        m = CLIPMultiTaskClassifier(INPUT_DIM, 2, 3, len(METADATA_COLUMNS), include_metadata=True)
        m.load_state_dict(state_dict)
        m.to(DEVICE).eval()
        _clip_model = m

def _to_three_way_probs(probs, task):
    """
    Map your outputs to dict with keys: Real, SatireMixed, Fake
    (2-way: 0=Fake, 1=Real) → (Real, SatireMixed=0, Fake)
    (3-way: 0=Real,1=Satire,2=Fake) → direct map
    """
    if task == "2way":
        p_fake = float(probs[0]); p_real = float(probs[1])
        return {"Real": p_real, "SatireMixed": 0.0, "Fake": p_fake}
    else:
        return {"Real": float(probs[0]), "SatireMixed": float(probs[1]), "Fake": float(probs[2])}

@torch.no_grad()
def run_clip(
    image_path_or_url: str,
    caption: str,
    task: str = "3way",
    metadata: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Returns: {"Real": p0, "SatireMixed": p1, "Fake": p2}
    """
    _lazy_load()

    md = metadata or {}  # 🔴 ADDED: safe guard if metadata is None

    # Build a one-row DataFrame so we can reuse your dataloader (ensures identical preprocessing)
    row = {
        "id": md.get("id", "single"),
        "image_url": image_path_or_url,          # your loaders use URL-aware preprocessing
        "clean_title": caption,

        # Include metadata columns (your loader expects these names)
        "num_comments": float(md.get("num_comments", 0) or 0),
        "score": float(md.get("score", 0) or 0),
        "upvote_ratio": float(md.get("upvote_ratio", 0) or 0),

        # 🔴 ADDED: DUMMY LABELS so your dataset doesn't KeyError
        # 2-way mapping: 0=Fake, 1=Real → pick 1 (Real) as harmless placeholder
        # 3-way mapping: 0=Real, 1=Satire/Mixed, 2=Fake → pick 0 (Real) as placeholder
        "2_way_label": 1,
        "3_way_label": 0,
    }
    df = pd.DataFrame([row])

    loader = prepare_dataloader(
        dataframe=df,
        processor=_clip_proc,
        metadata_columns=METADATA_COLUMNS,
        batch_size=1,
        shuffle=False,
        include_metadata=True
    )

    for batch in loader:
        # (Your existing code)
        if not batch:  
            print(f"[run_clip] Warning: empty batch for id={row['id']} (bad/missing image). Returning neutral fallback.")
            return {"Real": 0.33, "SatireMixed": 0.34, "Fake": 0.33}
        
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values = batch["pixel_values"].to(DEVICE)
        meta_t = batch["metadata"].to(DEVICE)

        out = _clip_model(input_ids, attention_mask, pixel_values, meta_t)
        logits = out["2_way"] if task == "2way" else out["3_way"]
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        return _to_three_way_probs(probs, task)

