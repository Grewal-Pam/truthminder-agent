# truthmindr_agent/tools/nli.py
from transformers import pipeline

_nli = pipeline("text-classification", model="roberta-large-mnli", truncation=True)


def safe_nli(ocr_text: str, caption: str):
    if not ocr_text:
        return {
            "label": "NEUTRAL_NO_OCR",
            "score": 1.0,
            "consistency_score": 0.5,
            "no_ocr": True,
        }
    out = _nli({"text": ocr_text, "text_pair": caption}, top_k=1)[0]
    lab = out["label"]
    cons = 1.0 if lab == "ENTAILMENT" else (0.5 if lab == "NEUTRAL" else 0.0)
    return {
        "label": lab,
        "score": float(out["score"]),
        "consistency_score": cons,
        "no_ocr": False,
    }
