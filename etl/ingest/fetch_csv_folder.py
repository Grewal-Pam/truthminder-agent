# etl/ingest/fetch_csv.py
import os, pandas as pd

def _detect_sep(path: str) -> str:
    p = path.lower()
    if p.endswith(".tsv") or p.endswith(".txt"):
        return "\t"
    return ","

def fetch_from_csv_folder(folder="data/raw/manual"):
    """
    Reads all CSV/TSV in the folder and maps columns to our schema.
    Expected columns: id, title, image_url, score, num_comments, upvote_ratio
    Adds source='csv'
    """
    if not os.path.isdir(folder):
        return pd.DataFrame(columns=[
            "id","title","image_url","score","num_comments","upvote_ratio","source"
        ])

    frames = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith((".csv",".tsv",".txt")):
            continue

        path = os.path.join(folder, fn)
        try:
            sep = _detect_sep(path)
            df = pd.read_csv(path, sep=sep)

            cols = {c.lower(): c for c in df.columns}

            def get(col, default):
                return df[cols[col]] if col in cols else default

            out = pd.DataFrame({
                "id": get("id", pd.Series([None]*len(df))),
                "title": get("title", pd.Series([""]*len(df))).fillna(""),
                "image_url": get("image_url", pd.Series([""]*len(df))).fillna(""),
                "score": pd.to_numeric(get("score", pd.Series([0]*len(df))), errors="coerce").fillna(0).astype(int),
                "num_comments": pd.to_numeric(get("num_comments", pd.Series([0]*len(df))), errors="coerce").fillna(0).astype(int),
                "upvote_ratio": pd.to_numeric(get("upvote_ratio", pd.Series([0.0]*len(df))), errors="coerce").fillna(0.0),
            })

            # synthesize missing IDs
            if out["id"].isna().any():
                out.loc[out["id"].isna(), "id"] = out.index[out["id"].isna()].map(lambda i: f"{fn}::row{i}")

            out["source"] = "csv"
            frames.append(out)

        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["id","title","image_url","score","num_comments","upvote_ratio","source"]
    )
