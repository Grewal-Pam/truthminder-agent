import os, csv, json, warnings

# Original imports
from models.clip_infer import run_clip
from models.vilt_infer import run_vilt
from models.flava_infer import run_flava
from tools.ocr import ocr
from tools.consistency import build_consistency_features
from tools.retrieval import retrieve_evidence

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TASK = os.environ.get("TRUTHMINDR_TASK", "3way")  # "2way" or "3way"
ABSTAIN_T = float(os.environ.get("TRUTHMINDR_ABSTAIN_T", "0.45"))

# ----------------------------
# LAZY LOAD MODELS WRAPPER
# ----------------------------
_models_cache = None

def load_models():
    """
    Loads heavy modules only once.
    We aren't rewriting infer functions yet — just caching references.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    _models_cache = {
        "clip_fn": run_clip,         
        "vilt_fn": run_vilt,
        "flava_fn": run_flava,
        "nli_fn": build_consistency_features,
        "retrieval_fn": retrieve_evidence,
    }
    return _models_cache


def arbiter(clip_p, vilt_p, flava_p, cons_feat, meta):
    labels = list(clip_p.keys())  # ["Real","SatireMixed","Fake"]
    avg = {L: (clip_p[L] + vilt_p[L] + flava_p[L]) / 3 for L in labels}
    delta = 0.06 * (cons_feat["consistency_score"] - 0.5)
    for L in avg:
        avg[L] += delta
    final = max(avg, key=avg.get)
    conf = float(avg[final])
    abstain = (conf < ABSTAIN_T) or (
        cons_feat["nli"]["label"] == "CONTRADICTION" and final == "Real"
    )
    return ("ABSTAIN" if abstain else final), conf


def run_row(row: dict, models=None):
    if models is None:
        models = load_models()

    trace = {"post_id": row["id"], "steps": []}

    # 1) Perception
    clip_p = models["clip_fn"](row["image_url"], row["clean_title"], task=TASK, metadata=row)
    vilt_p = models["vilt_fn"](row["image_url"], row["clean_title"], task=TASK, metadata=row)
    flava_p = models["flava_fn"](row["image_url"], row["clean_title"], task=TASK, metadata=row)

    trace["steps"].append({"stage": "perception", "clip": clip_p, "vilt": vilt_p, "flava": flava_p})

    # 2) Evidence (OCR)
    ocr_text = ocr(row["image_url"])
    trace["steps"].append({"stage": "evidence", "ocr_text": ocr_text, "no_ocr": (ocr_text == "")})

    # 3) Consistency (NLI + clip-text cosine)
    cons_feat = models["nli_fn"](row["image_url"], row["clean_title"], ocr_text)
    trace["steps"].append({"stage": "consistency", **cons_feat})

    # 4) Retrieval (RAG)
    query_text = f"{row['clean_title']} {ocr_text}"
    retrieved_text = models["retrieval_fn"](query_text)
    trace["steps"].append({"stage":"retrieval", "retrieved_evidence": retrieved_text})

    # 5) Arbiter
    final_label, final_conf = arbiter(clip_p, vilt_p, flava_p, cons_feat, row)

    result = {
        "post_id": row["id"],
        "final_label": final_label,
        "final_confidence": final_conf,
        "nli_label": cons_feat["nli"]["label"],
        "consistency_score": cons_feat["consistency_score"],
        "clip_cos": cons_feat["clip_cos"],
        "ocr_text": (ocr_text or "")[:180],
        "upvote_ratio": row.get("upvote_ratio", ""),
        "score": row.get("score", ""),
        "num_comments": row.get("num_comments", ""),
        "retrieved_text": retrieved_text,
    }
    trace["final"] = result
    return result, trace


def _detect_sep(path):
    return "\t" if path.endswith(".tsv") or path.endswith(".tsv.gz") else ","


def run_file(path="data/test_batch_1.tsv", out_dir="outputs"):
    models = load_models()  # Ensure one-time load
    os.makedirs(f"{out_dir}/traces", exist_ok=True)
    enriched = []
    sep = _detect_sep(path)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            res, tr = run_row(row, models=models)
            enriched.append({**row, **res})
            with open(f'{out_dir}/traces/{row["id"]}.json', "w", encoding="utf-8") as w:
                json.dump(tr, w, ensure_ascii=False, indent=2)

    out_csv = os.path.join(out_dir, "enriched.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as w:
        fieldnames = enriched[0].keys()
        wr = csv.DictWriter(w, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(enriched)
    print(f"[DONE] Wrote {out_csv} and traces/*.json")


if __name__ == "__main__":
    run_file()
