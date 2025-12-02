import pandas as pd
import requests, os, json
from pathlib import Path
from datetime import datetime

def fetch_reddit_posts(limit=50, subreddit="worldnews"):
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}"
    headers = {"User-agent": "TruthMindr/1.0"}
    res = requests.get(url, headers=headers, timeout=10)
    data = res.json()

    posts = []
    for post in data.get("data", {}).get("children", []):
        d = post["data"]
        posts.append({
            "id": d["id"],
            "title": d["title"],
            "clean_title": d["title"],  # placeholder for later NLP cleaning
            "image_url": d.get("url_overridden_by_dest", ""),
            "score": d["score"],
            "num_comments": d["num_comments"],
            "upvote_ratio": d.get("upvote_ratio", 0.0),
            "source": "reddit"
        })

    return pd.DataFrame(posts)


if __name__ == "__main__":
    # ✅ Always save relative to project root
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_DIR = BASE_DIR / "data" / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_reddit_posts()

    # ✅ create filename
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    file_path = RAW_DIR / f"reddit_{ts}.json"

    # ✅ save JSON
    df.to_json(file_path, orient="records", indent=2)

    print(f"✅ Saved {len(df)} reddit posts → {file_path}")
