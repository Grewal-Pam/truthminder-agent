import duckdb
import pandas as pd

def save_to_duckdb(df):
    con = duckdb.connect("truthmindr.db")

    # ✅ Create correct final schema (9 columns)
    con.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        title TEXT,
        image_url TEXT,
        score DOUBLE,
        num_comments DOUBLE,
        upvote_ratio DOUBLE,
        title_length DOUBLE,
        image_available BOOLEAN,
        source TEXT
    );
    """)

    # ✅ Ensure df has exactly these columns
    expected_cols = [
        "id", "title", "image_url", "score", "num_comments",
        "upvote_ratio", "title_length", "image_available", "source"
    ]

    # Add missing columns to DF to prevent errors
    for col in expected_cols:
        if col not in df.columns:
            if col == "source":
                df[col] = "unknown"
            elif col in ["title_length", "num_comments", "score"]:
                df[col] = 0
            elif col in ["image_available"]:
                df[col] = False
            else:
                df[col] = ""

    # ✅ Reorder DF to match table schema
    df = df[expected_cols]

    # ✅ Create temp table and insert
    con.execute("CREATE TEMP TABLE incoming AS SELECT * FROM df")

    con.execute("""
        INSERT OR REPLACE INTO posts
        SELECT * FROM incoming;
    """)

    print(f"✅ Loaded {len(df)} rows into DuckDB (posts table)")
