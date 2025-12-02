# 🔄 TruthMindr Airflow DAGs

Apache Airflow orchestration for automating the TruthMindr ETL pipeline with production-grade scheduling, monitoring, and error handling.

---

## 📋 DAG Overview

### `truthmindr_etl` DAG
**Purpose:** Daily automated data ingestion, transformation, and enrichment

| Property | Value |
|----------|-------|
| **DAG ID** | `truthmindr_etl` |
| **Schedule** | Daily at 00:00 UTC (`0 0 * * *`) |
| **Owner** | truthmindr |
| **Retries** | 2 attempts per task |
| **Backfill** | Disabled |

---

## 📊 Pipeline Tasks

```
check_environment
      ↓
run_etl_pipeline
      ↓
verify_data_loaded
```

### Task Details:

#### 1️⃣ **check_environment**
- **Type:** PythonOperator
- **Purpose:** Verify required environment variables (NEWSAPI_KEY)
- **Action:** Fails gracefully with warnings if key is missing
- **Duration:** ~5 seconds

#### 2️⃣ **run_etl_pipeline**
- **Type:** BashOperator
- **Purpose:** Execute the main ETL workflow
- **Executes:**
  - Ingest from Reddit (r/worldnews)
  - Ingest from NewsAPI (top headlines)
  - Ingest from CSV folder (manual uploads)
  - Clean and normalize text
  - Enrich with metadata
  - Generate ML predictions (CLIP, ViLT, FLAVA)
  - Load into DuckDB and Parquet
- **Duration:** ~5-15 minutes (depends on source availability)
- **Retries:** 2 with 10-minute delays

#### 3️⃣ **verify_data_loaded**
- **Type:** BashOperator
- **Purpose:** Validate successful data load
- **Output:** DuckDB query results showing posts per source
- **Duration:** ~10 seconds

---

## 🚀 Setup & Installation

### Prerequisites:
```bash
# 1. Install Airflow (if not already installed)
pip install apache-airflow==2.7.0

# 2. Initialize Airflow database
airflow db init

# 3. Create default Airflow user
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

### Deploy the DAG:

```bash
# 1. Copy DAG to Airflow DAGs directory
cp airflow_dags/truthmindr_etl_dag.py ~/airflow/dags/

# 2. Set required environment variables
export NEWSAPI_KEY="YOUR_NEWSAPI_KEY_HERE"

# 3. Start Airflow scheduler (background)
nohup airflow scheduler > airflow_scheduler.log 2>&1 &

# 4. Start Airflow webserver (port 8080)
airflow webserver --port 8080

# 5. Access UI at http://localhost:8080
```

---

## 📈 Monitoring & Debugging

### View DAG Status:
```bash
# List all DAGs
airflow dags list

# Check DAG structure
airflow dags show truthmindr_etl

# List all runs of the DAG
airflow dags list-runs -d truthmindr_etl
```

### Monitor Task Execution:
```bash
# View Airflow logs
tail -f ~/airflow/logs/truthmindr_etl/*.log

# Trigger manual run
airflow dags trigger -e 2025-12-02 truthmindr_etl

# View task status
airflow tasks list truthmindr_etl
```

### Access Web UI:
1. Open **http://localhost:8080** in browser
2. Login with credentials (default: admin/admin)
3. Navigate to **DAGs** → **truthmindr_etl**
4. View:
   - Task execution timeline
   - Logs for each task
   - Data lineage and relationships
   - Retry history

---

## ⚙️ Configuration

### Schedule Interval Options:

```python
# Daily at 00:00 UTC
schedule_interval='0 0 * * *'

# Every 6 hours
schedule_interval='0 */6 * * *'

# Weekly on Monday at 00:00
schedule_interval='0 0 * * MON'

# Cron expression (standard format)
schedule_interval='30 2 * * *'  # 02:30 UTC

# Predefined shortcuts
schedule_interval='@daily'      # Daily
schedule_interval='@weekly'     # Weekly
schedule_interval='@monthly'    # Monthly
```

### Adjust in DAG:
```python
with DAG(
    dag_id='truthmindr_etl',
    ...
    schedule_interval='0 2 * * *',  # Change to 02:00 UTC
    ...
) as dag:
```

---

## 🔧 Troubleshooting

### Issue: DAG not appearing in Airflow UI
```bash
# Verify DAG file is in correct location
ls ~/airflow/dags/truthmindr_etl_dag.py

# Refresh DAG list
airflow dags trigger truthmindr_etl

# Check Airflow logs for parsing errors
tail -f ~/airflow/logs/dag_processor_manager/*.log
```

### Issue: NewsAPI task fails
```bash
# Verify NEWSAPI_KEY is set
echo $NEWSAPI_KEY

# Set in Airflow (Alternative):
airflow variables set NEWSAPI_KEY "your_key_here"
```

### Issue: Memory error during ML enrichment
```bash
# Reduce batch size in ETL
# Edit etl/pipeline.py and set limit=50
```

### Issue: Database lock error
```bash
# Kill any existing DuckDB connections
pkill -f duckdb

# Or remove lock file
rm -f truthmindr.db.wal
```

---

## 📊 Performance Tuning

### Parallel Task Execution:
```python
# Run up to 4 tasks in parallel
max_active_tasks = 4
```

### Timeout Configuration:
```python
default_args = {
    'execution_timeout': timedelta(minutes=30),  # Kill if exceeds 30 min
    'retry_delay': timedelta(minutes=5),
}
```

### SLA (Service Level Agreement):
```python
default_args = {
    'sla': timedelta(minutes=30),  # Alert if exceeds 30 min
}
```

---

## 🚀 Production Deployment

### For Production Environments:

1. **Use External Database** (PostgreSQL instead of SQLite):
```bash
export AIRFLOW__CORE__SQL_ALCHEMY_CONN='postgresql://user:pass@localhost/airflow'
```

2. **Enable Authentication:**
```python
# In airflow.cfg
[api]
auth_backend = airflow.api.auth.backend.basic_auth
```

3. **Setup Logging** to cloud storage:
```bash
# Log to S3
export AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER='s3://your-bucket/airflow-logs'
```

4. **Configure Alerts:**
```python
default_args = {
    'email': ['admin@truthmindr.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}
```

---

## 📚 References

- [Apache Airflow Documentation](https://airflow.apache.org/)
- [Airflow DAG Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
- [Cron Expression Reference](https://crontab.guru/)
- [TruthMindr ETL Pipeline](../etl/README.md)

---

## 🔗 Integration

- **ETL Pipeline:** See `etl/README.md`
- **Data Schema:** See `etl/README.md#output-schema`
- **Main Agent:** See `agent/runner.py`
