import os, time, pandas as pd, requests, json
from pathlib import Path
from datetime import datetime

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"

def fetch_newsapi_headlines(country="us", page_size=50, pages=1):
    """
    Requires: export NEWSAPI_KEY="2f55ca7a72e044a0b2511359019cdb85"
    Returns unified columns for ETL.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing NEWSAPI_KEY env var")

    rows = []
    for page in range(1, pages + 1):
        params = {"country": country, "pageSize": page_size, "page": page, "apiKey": api_key}
        r = requests.get(NEWSAPI_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for a in data.get("articles", []):
            # Create deterministic ID
            aid = f"{(a.get('source') or {}).get('id') or 'na'}::{a.get('publishedAt','na')}::{hash(a.get('title',''))}"

            rows.append({
                "id": aid,
                "title": a.get("title",""),
                "clean_title": a.get("title",""),
                "image_url": a.get("urlToImage") or "",
                "score": 0,
                "num_comments": 0,
                "upvote_ratio": 0.0,
                "source": "newsapi"
            })

        time.sleep(0.2)  # polite API pacing

    return pd.DataFrame(rows)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_DIR = BASE_DIR / "data" / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_newsapi_headlines()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    file_path = RAW_DIR / f"newsapi_{ts}.json"

    df.to_json(file_path, orient="records", indent=2)

    print(f"✅ Saved {len(df)} NewsAPI articles → {file_path}")
