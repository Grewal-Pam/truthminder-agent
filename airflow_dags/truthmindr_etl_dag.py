from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'truthmindr',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='truthmindr_etl',
    default_args=default_args,
    description='Trigger TruthMindr Prefect ETL',
    schedule_interval='@daily',  # You can update later
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    run_prefect_etl = BashOperator(
        task_id='run_prefect_etl',
        bash_command="""
        cd ~/projects/project && \
        conda activate myenv && \
        prefect flow run --name truthmindr-etl-flow
        """,
    )

    run_prefect_etl
