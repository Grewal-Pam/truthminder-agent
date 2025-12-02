# analytics_sql/scripts/01_duckdb_basics.py
# Run: python analytics_sql/scripts/01_duckdb_basics.py

import duckdb
from pathlib import Path

# 1️⃣ Locate your actual dataset
# Your file path: analytics_sql/enriched_sample.csv
ROOT = Path(__file__).resolve().parents[2]  # repo root
csv_path = ROOT / "analytics_sql" / "enriched_sample.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")

print(f"✅ Using dataset: {csv_path}\n")

# 2️⃣ Connect (in-memory)
con = duckdb.connect()

# 3️⃣ Sanity check
df = con.execute("SELECT 1+1 AS result").fetchdf()
print("Sanity check:\n", df, "\n")

# 4️⃣ Create a view over your CSV (DuckDB can auto-detect CSV/TSV delimiters)
# If your file is comma-separated, no need for delim='\t'
con.execute(f"""
    CREATE OR REPLACE VIEW fakeddit AS
    SELECT *
    FROM read_csv_auto('{csv_path.as_posix()}', header=True);
""")

# 5️⃣ BASIC QUERIES (Lesson 1)
print("Total rows:")
print(con.execute("SELECT COUNT(*) AS total_rows FROM fakeddit;").fetchdf(), "\n")

print("Unique subreddits:")
print(con.execute("SELECT COUNT(DISTINCT subreddit) AS unique_subreddits FROM fakeddit;").fetchdf(), "\n")

print("Average upvote ratio:")
print(con.execute("SELECT ROUND(AVG(upvote_ratio), 3) AS avg_upvote_ratio FROM fakeddit;").fetchdf(), "\n")

print("Label distribution (2-way):")
print(con.execute("""
    SELECT "2_way_label" AS label_2way,
           COUNT(*) AS posts,
           ROUND(AVG(upvote_ratio), 3) AS avg_ratio
    FROM fakeddit
    GROUP BY 1
    ORDER BY 1;
""").fetchdf(), "\n")

print("Top 10 domains by average score (min 5 posts):")
print(con.execute("""
    SELECT domain,
           COUNT(*) AS posts,
           ROUND(AVG(score), 3) AS avg_score
    FROM fakeddit
    GROUP BY domain
    HAVING COUNT(*) >= 5
    ORDER BY avg_score DESC
    LIMIT 10;
""").fetchdf(), "\n")

# 6️⃣ Save a small result for Power BI / Excel (optional)
OUT_DIR = ROOT / "analytics_sql" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

con.execute("""
    COPY (
        SELECT subreddit,
               COUNT(*) AS posts,
               ROUND(AVG(upvote_ratio), 3) AS avg_ratio
        FROM fakeddit
        GROUP BY subreddit
        ORDER BY posts DESC
        LIMIT 50
    ) TO ?
    WITH (FORMAT CSV, HEADER TRUE);
""", [str(OUT_DIR / "subreddit_overview.csv")])

print(f"💾 Saved summary: {OUT_DIR / 'subreddit_overview.csv'}")
