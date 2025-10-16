# truthmindr_agent/tools/consistency.py
import io, requests, torch
from PIL import Image
from typing import Optional
from transformers import CLIPProcessor, CLIPModel

_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device).eval()
_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def _open_local_or_url(path_or_url: str):
    if path_or_url.startswith("http"):
        try:
            r = requests.get(path_or_url, timeout=5); r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None
    else:
        try:
            return Image.open(path_or_url).convert("RGB")
        except Exception:
            return None

@torch.no_grad()
def clip_image_text_cosine(path_or_url: str, caption: str) -> Optional[float]:
    img = _open_local_or_url(path_or_url)
    if img is None:
        return None
    inputs = _proc(text=[caption], images=[img], return_tensors="pt", padding=True).to(_device)
    out = _clip(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    cos = (img_emb @ txt_emb.T).squeeze().item()
    return 0.5*(cos+1.0)  # [-1,1] → [0,1]

def build_consistency_features(path_or_url: str, caption: str, ocr_text: str):
    from .nli import safe_nli
    nli_out = safe_nli(ocr_text, caption)
    clip_cos = clip_image_text_cosine(path_or_url, caption)
    if clip_cos is None:
        # no image access; rely only on NLI
        score = nli_out["consistency_score"]
        clip_cos_val = ""
    else:
        if nli_out["no_ocr"]:
            score = 0.6*clip_cos + 0.4*0.5
        else:
            score = 0.7*nli_out["consistency_score"] + 0.3*clip_cos
        clip_cos_val = float(clip_cos)
    return {"nli": nli_out, "clip_cos": clip_cos_val, "consistency_score": float(score)}
