import pandas as pd
import os
from datetime import datetime

BRONZE_DIR = "data/lake/bronze"
SILVER_DIR = "data/lake/silver"

def load_bronze():
    frames = []
    for root, _, files in os.walk(BRONZE_DIR):
        for f in files:
            if f.endswith(".parquet"):
                frames.append(pd.read_parquet(os.path.join(root, f)))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def clean_silver(df):
    # Standardizing schema
    df = df.rename(columns={
        "title_length": "title_len",
        "image_available": "has_image"
    })

    # Ensure types
    df["title"] = df["title"].astype(str).fillna("")
    df["clean_title"] = df["clean_title"].astype(str).fillna("")
    df["image_url"] = df["image_url"].astype(str).fillna("")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).astype(int)
    df["upvote_ratio"] = pd.to_numeric(df["upvote_ratio"], errors="coerce").fillna(0.0)
    df["source"] = df["source"].astype(str)

    # Add column: text length
    df["text_length"] = df["clean_title"].apply(len)

    # Drop duplicates
    df = df.drop_duplicates(subset=["id","source"])

    return df


def save_silver(df):
    ts = datetime.utcnow().strftime("%Y%m%d")
    path = f"{SILVER_DIR}/date={ts}"
    os.makedirs(path, exist_ok=True)
    file = f"{path}/silver.parquet"

    df.to_parquet(file, index=False)
    print(f"💿 [silver] wrote {file}")


def generate_silver():
    bronze_df = load_bronze()
    if bronze_df.empty:
        print("⚠️ No bronze data found.")
        return
    
    silver = clean_silver(bronze_df)
    save_silver(silver)
    print(f"✅ Silver layer ready: {len(silver)} rows")
