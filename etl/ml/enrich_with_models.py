import pandas as pd
import os, time
from datetime import datetime
from agent.runner import load_models, run_row

SILVER_DIR = "data/lake/silver"
GOLD_DIR = "data/lake/gold"

def get_latest_silver():
    dates = sorted(os.listdir(SILVER_DIR))
    latest = dates[-1]
    return os.path.join(SILVER_DIR, latest, "silver.parquet")

def run_inference(df):
    models = load_models()
    results = []

    for _, row in df.iterrows():
        r = {
            "id": row["id"],
            "source": row["source"],
            "title": row["title"],
            "clean_title": row["clean_title"],
            "image_url": row["image_url"],
            "score": row["score"],
            "num_comments": row["num_comments"],
            "upvote_ratio": row["upvote_ratio"],
        }

        result, trace = run_row(row.to_dict(), models=models)

        r["final_label"] = result["final_label"]
        r["final_confidence"] = result["final_confidence"]
        r["consistency_score"] = result["consistency_score"]
        r["nli_label"] = result["nli_label"]

        # Extract model raw probs
        step0 = trace["steps"][0]
        r["clip_real"]  = step0["clip"]["Real"]
        r["clip_fake"]  = step0["clip"]["Fake"]
        r["vilt_real"]  = step0["vilt"]["Real"]
        r["vilt_fake"]  = step0["vilt"]["Fake"]
        r["flava_real"] = step0["flava"]["Real"]
        r["flava_fake"] = step0["flava"]["Fake"]

        r["timestamp"] = datetime.utcnow().isoformat()
        results.append(r)

    return pd.DataFrame(results)

def save_gold(df):
    ts = datetime.utcnow().strftime("%Y%m%d")
    path = f"{GOLD_DIR}/date={ts}"
    os.makedirs(path, exist_ok=True)
    file = f"{path}/gold.parquet"
    df.to_parquet(file, index=False)
    print(f"🏅 [gold] wrote {file}")

def generate_gold():
    silver = get_latest_silver()
    df = pd.read_parquet(silver)
    gold = run_inference(df)
    save_gold(gold)
    print(f"✅ Gold layer ready: {len(gold)} rows")
