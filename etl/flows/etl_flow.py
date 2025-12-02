from prefect import flow, task
from etl.pipeline import run
import datetime

@task(name="Run TruthMindr ETL")
def run_etl_task(limit: int = 50):
    run(limit=limit)

@flow(name="truthmindr-etl-flow")
def truthmindr_etl_flow(limit: int = 50):
    print(f"🚀 Prefect Flow Triggered at {datetime.datetime.now()}")
    run_etl_task(limit)
    print("✅ Prefect ETL Flow completed")

if __name__ == "__main__":
    truthmindr_etl_flow()
