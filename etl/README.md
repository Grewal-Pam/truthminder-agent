## New sources
- **Reddit**: r/worldnews top posts
- **NewsAPI**: top headlines (set `NEWSAPI_KEY` in environment)
- **CSV Folder**: drop any CSV/TSV into `data/raw/` and it will be ingested

### Environment
```bash
export NEWSAPI_KEY="YOUR_KEY"

## `Makefile` (append targets)
```make
ingest:
\tpython -m etl.pipeline

ingest_us:
\tNEWSAPI_KEY=$$NEWSAPI_KEY python -m etl.pipeline

show_posts:
\tpython - << 'EOF'\nimport duckdb\ncon=duckdb.connect('truthmindr.db')\nprint(con.execute(\"select source, count(*) from posts group by source\").fetchdf())\nprint(con.execute(\"select * from posts order by rowid desc limit 5\").fetchdf())\ncon.close()\nEOF

# 1) set your key once (bash)
export NEWSAPI_KEY="YOUR_NEWSAPI_KEY"

# 2) run pipeline (all sources)
python etl/pipeline.py

# 3) inspect
python - << 'EOF'
import duckdb
con=duckdb.connect('truthmindr.db')
print(con.execute("select source, count(*) from posts group by source").fetchdf())
print(con.execute("select * from posts limit 5").fetchdf())
con.close()
EOF
