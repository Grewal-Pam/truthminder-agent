"""
TruthMindr ETL Orchestration DAG for Apache Airflow

This DAG triggers the daily ETL pipeline that:
1. Ingests posts from Reddit, NewsAPI, and CSV folders
2. Cleans and normalizes text
3. Enriches with metadata features
4. Generates model predictions (CLIP, ViLT, FLAVA)
5. Loads enriched data into DuckDB and Parquet

Schedule: Daily at 00:00 UTC
Owner: TruthMindr Team
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

# Environment setup
def setup_env(**context):
    """Verify required environment variables for ETL pipeline"""
    required_vars = ['NEWSAPI_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("   Set them with: export NEWSAPI_KEY='your_key'")
    else:
        print("✅ All required environment variables are set")

# Default arguments for all tasks
default_args = {
    'owner': 'truthmindr',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
}

# Define the DAG
with DAG(
    dag_id='truthmindr_etl',
    default_args=default_args,
    description='Daily ETL pipeline: Ingest, Transform, Enrich, Load multimodal disinformation data',
    schedule_interval='0 0 * * *',  # Daily at 00:00 UTC
    catchup=False,
    tags=['etl', 'truthmindr', 'data-pipeline'],
) as dag:

    # Task 1: Verify environment setup
    check_env = PythonOperator(
        task_id='check_environment',
        python_callable=setup_env,
        doc='Verify NEWSAPI_KEY and other required environment variables',
    )

    # Task 2: Run the ETL pipeline
    run_etl = BashOperator(
        task_id='run_etl_pipeline',
        bash_command="""
        set -e
        cd /home/ubuntu/projects/project && \
        source activate myenv && \
        echo "🚀 Starting TruthMindr ETL Pipeline..." && \
        python -m etl.pipeline && \
        echo "✅ ETL completed successfully"
        """,
        doc='Execute ETL pipeline: fetch from Reddit/NewsAPI/CSV → clean → enrich → load',
        retries=2,
    )

    # Task 3: Verify data loaded
    verify_data = BashOperator(
        task_id='verify_data_loaded',
        bash_command="""
        cd /home/ubuntu/projects/project && \
        source activate myenv && \
        python - << 'EOF'
import duckdb
con = duckdb.connect('truthmindr.db')
result = con.execute("SELECT source, COUNT(*) as count FROM posts GROUP BY source").fetchdf()
print("📊 Posts loaded by source:")
print(result)
total = con.execute("SELECT COUNT(*) as total FROM posts").fetchone()[0]
print(f"\n✅ Total posts in database: {total}")
con.close()
EOF
        """,
        doc='Verify that data was successfully loaded into DuckDB',
    )

    # Define task dependencies
    check_env >> run_etl >> verify_data

