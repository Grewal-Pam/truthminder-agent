# truthmindr_agent/models/vilt_infer.py
import os, glob, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from transformers import ViltProcessor, ViltModel
from models.vilt_model import ViltClassificationModel
from dataloaders.vilt_dataloader import get_vilt_dataloader
from typing import Optional, Dict, Any


DEVICE = os.environ.get("TRUTHMINDR_DEVICE", "cpu")
METADATA_COLUMNS = ["num_comments", "score", "upvote_ratio"]
# Folder with many files; we'll pick 3-way + with_metadata by pattern:
VILT_RUN_DIR = "runs/VILT/2025-04-06_16-48-46_GOOD/models"

_vilt_proc = None
_vilt_base = None
_vilt_model = None

def _pick_checkpoint(num_labels=3, include_metadata=True):
    tag = "with_metadata" if include_metadata else "without_metadata"
    # Your naming: vilt_final_model_{task_name}_{tag}.pth, where task_name has '3-way' or '3_way'
    pat1 = os.path.join(VILT_RUN_DIR, f"vilt_final_model_3-way_classification_{tag}.pth")
    pat2 = os.path.join(VILT_RUN_DIR, f"vilt_final_model_3_way_classification_{tag}.pth")
    pat3 = os.path.join(VILT_RUN_DIR, f"*3*classification*{tag}*.pth")
    cands = [p for p in [pat1, pat2] if os.path.exists(p)] or glob.glob(pat3)
    if not cands:
        raise FileNotFoundError("No ViLT 3-way with_metadata checkpoint found. Please provide exact filename.")
    # newest by mtime
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _lazy_load():
    global _vilt_proc, _vilt_base, _vilt_model
    if _vilt_proc is None:
        _vilt_base = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        _vilt_proc = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        ckpt = _pick_checkpoint(num_labels=3, include_metadata=True)
        m = ViltClassificationModel(
            vilt_model=_vilt_base,
            num_labels=3,
            metadata_dim=len(METADATA_COLUMNS),
            include_metadata=True
        )
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.to(DEVICE).eval()
        _vilt_model = m

def _to_three_way_probs(logits, task):
    if task == "2way":
        # logits = [Fake, Real] by your mapping
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        p_fake, p_real = float(probs[0]), float(probs[1])
        return {"Real": p_real, "SatireMixed": 0.0, "Fake": p_fake}
    else:
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()  # [Real, Satire, Fake]
        return {"Real": float(probs[0]), "SatireMixed": float(probs[1]), "Fake": float(probs[2])}

@torch.no_grad()
def run_vilt(image_path_or_url: str, caption: str, task: str = "3way", metadata: Optional[Dict[str, Any]] = None) -> dict:
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

    # Your vilt dataloader returns (loader, something). We need loader only.
    loader, _ = get_vilt_dataloader(
        df,
        _vilt_proc,
        label_type="3_way_label" if task == "3way" else "2_way_label",
        batch_size=1,
        shuffle=False,
        metadata_columns=METADATA_COLUMNS
    )

    for batch in loader:
        if not batch:
            print(f"[run_vilt] Warning: empty batch for id={row['id']} (bad/missing image). Returning neutral fallback.")
            return {"Real": 0.33, "SatireMixed": 0.34, "Fake": 0.33}
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values = batch["pixel_values"].to(DEVICE)
        meta_t = batch["metadata"].to(DEVICE)

        logits = _vilt_model(input_ids, attention_mask, pixel_values, meta_t)
        return _to_three_way_probs(logits, task)

