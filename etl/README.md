# 🔄 TruthMindr ETL Pipeline

A modular Extract-Transform-Load (ETL) system for ingesting, cleaning, and enriching multimodal disinformation data from multiple sources.

---

## 📊 Data Sources

### Supported Ingest Channels:
1. **Reddit** - r/worldnews top posts via PRAW API
2. **NewsAPI** - Top headlines by country (requires `NEWSAPI_KEY`)
3. **CSV/TSV Folder** - Manual data drops into `data/raw/manual/`

---

## 🏗️ Pipeline Architecture

The ETL follows a **layered medallion architecture**:

```
[Ingest Layer]
    ↓
[Transform Layer] → Clean Text + Enrich Metadata + Deduplicate
    ↓
[Load Layer] → DuckDB + Parquet (Bronze → Silver → Gold)
    ↓
[ML Enrichment] → Add model predictions
```

### Pipeline Stages:

| Stage | Module | Description |
|-------|--------|-------------|
| **Ingest** | `ingest/fetch_*.py` | Fetch from Reddit, NewsAPI, or CSV folder |
| **Transform** | `transform/clean_text.py` | Tokenize, lowercase, remove URLs/special chars |
| **Enrich** | `transform/enrich_metadata.py` | Add timestamps, normalize scores, compute derived features |
| **Deduplicate** | `pipeline.py` | Remove duplicates on (id, source) |
| **Partition** | `load/to_parquet.py` | Store by source in Bronze layer |
| **Silver Layer** | `transform/to_silver.py` | Standardized schema across sources |
| **Gold Layer** | `ml/enrich_with_models.py` | Predictions from CLIP, ViLT, FLAVA |
| **Store** | `load/to_duckdb.py` | Load final data into DuckDB |

---

## 🚀 Quick Start

### 1. Set Environment Variables
```bash
export NEWSAPI_KEY="YOUR_NEWSAPI_KEY_HERE"
```

### 2. Run the Pipeline
```bash
# Ingest 50 posts from each source
python -m etl.pipeline

# Or with custom limits
python -c "from etl.pipeline import run; run(limit=100, country='gb')"
```

### 3. Verify Data Loaded
```bash
python - << 'EOF'
import duckdb
con = duckdb.connect('truthmindr.db')
print("📊 Source distribution:")
print(con.execute("SELECT source, COUNT(*) as count FROM posts GROUP BY source").fetchdf())
print("\n📝 Recent posts:")
print(con.execute("SELECT id, title, source FROM posts ORDER BY rowid DESC LIMIT 5").fetchdf())
con.close()
EOF
```

---

## 📁 Module Structure

```
etl/
├── __init__.py              # ETL module marker
├── pipeline.py              # Main orchestration (run function)
├── ingest/
│   ├── fetch_reddit.py      # Reddit API client
│   ├── fetch_newsapi.py     # NewsAPI client
│   └── fetch_csv_folder.py  # CSV/TSV directory reader
├── transform/
│   ├── clean_text.py        # Text normalization
│   ├── enrich_metadata.py   # Feature engineering
│   └── to_silver.py         # Standardization layer
├── load/
│   ├── to_duckdb.py         # DuckDB writer
│   └── to_parquet.py        # Parquet partitioner
├── ml/
│   └── enrich_with_models.py # Model predictions (CLIP, ViLT, FLAVA)
├── flows/
│   └── etl_flow.py          # Prefect flow definition
├── cleanup.sh               # Clear cached data
└── README.md                # This file
```

---

## ⚙️ Configuration

### Environment Variables:
- `NEWSAPI_KEY` - Required for NewsAPI source
- `PRAW_CLIENT_ID`, `PRAW_CLIENT_SECRET` - Optional: Reddit API credentials (uses public API by default)

### Pipeline Parameters:
```python
from etl.pipeline import run

# Ingest 100 posts per source from US
run(limit=100, country="us")

# Ingest 200 posts from UK
run(limit=200, country="gb")
```

---

## 📊 Output Schema

Final DuckDB table (`posts`) contains:

| Column | Type | Source |
|--------|------|--------|
| `id` | VARCHAR | Post ID (unique per source) |
| `title` | VARCHAR | Post headline |
| `clean_title` | VARCHAR | Normalized title |
| `image_url` | VARCHAR | Image URL (if available) |
| `source` | VARCHAR | 'reddit', 'newsapi', 'csv' |
| `timestamp` | TIMESTAMP | Publication date |
| `upvote_ratio` | FLOAT | Normalized engagement metric |
| `score` | FLOAT | Post score/votes |
| `num_comments` | INT | Comment count |
| `clip_pred` | VARCHAR | CLIP prediction (if enriched) |
| `vilt_pred` | VARCHAR | ViLT prediction (if enriched) |
| `flava_pred` | VARCHAR | FLAVA prediction (if enriched) |

---

## 🔧 Troubleshooting

### NewsAPI returns empty results:
- Verify `NEWSAPI_KEY` is set: `echo $NEWSAPI_KEY`
- Check API quota: [newsapi.org/account](https://newsapi.org/account)

### Reddit ingest fails:
- Public API doesn't require credentials
- Check network connectivity

### Out of memory during ML enrichment:
- Reduce post limit: `run(limit=50)`
- Process in batches manually

### Clear cached data:
```bash
bash etl/cleanup.sh
```

---

## 🚀 Integration with Airflow

See `airflow_dags/truthmindr_etl_dag.py` for scheduled daily runs.
