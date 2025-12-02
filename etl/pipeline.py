import os, json, pandas as pd
from datetime import datetime

from etl.ingest.fetch_reddit import fetch_reddit_posts
from etl.ingest.fetch_newsapi import fetch_newsapi_headlines
from etl.ingest.fetch_csv_folder import fetch_from_csv_folder

from etl.transform.clean_text import clean_text
from etl.transform.enrich_metadata import enrich_metadata
from etl.load.to_duckdb import save_to_duckdb

from etl.load.to_parquet import save_to_parquet

from etl.transform.to_silver import generate_silver

from etl.ml.enrich_with_models import generate_gold

RAW_DIR = "data/raw"

def _dump_raw(name: str, df: pd.DataFrame):
    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RAW_DIR, f"{name}_{ts}.jsonl")
    with open(path, "w", encoding="utf-8") as w:
        for _, row in df.iterrows():
            w.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[raw] wrote {path}")

def run(limit=50, country="us"):
    print("🚀 TruthMindr ETL started...")

    dfs = []

    # 1) Reddit
    try:
        r_df = fetch_reddit_posts(limit=limit, subreddit="worldnews")
        r_df["source"] = "reddit"
        dfs.append(r_df)
        _dump_raw("reddit_worldnews", r_df)
    except Exception as e:
        print("[WARN] reddit ingest failed:", e)

    # 2) NewsAPI
    try:
        n_df = fetch_newsapi_headlines(country=country, page_size=min(limit, 50), pages=max(1, limit//50))
        dfs.append(n_df)
        _dump_raw("newsapi_top", n_df)
    except Exception as e:
        print("[WARN] newsapi ingest failed:", e)

    # 3) CSV folder (user drop)
    try:
        c_df = fetch_from_csv_folder(folder="data/raw/manual")
        if len(c_df) > 0:
            dfs.append(c_df)
            _dump_raw("csv_folder_ingest", c_df)
    except Exception as e:
        print("[WARN] csv-folder ingest failed:", e)

    if not dfs:
        print("No data ingested."); return

    df = pd.concat(dfs, ignore_index=True)

    # clean + enrich
    df["clean_title"] = df["title"].astype(str).apply(clean_text)
    df = enrich_metadata(df)

    # dedupe on (id, source)
    df = df.drop_duplicates(subset=["id","source"])

    # Partition parquet by source
    for src in df["source"].unique():
        save_to_parquet(df[df["source"] == src], layer="bronze", source=src)

    generate_silver()
    generate_gold()
    save_to_duckdb(df)
    print(f"✅ ETL completed: {len(df)} rows loaded")

if __name__ == "__main__":
    run()
