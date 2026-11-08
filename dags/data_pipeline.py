"""DAG 1: daily Medallion ETL (Bronze -> Silver -> Gold)."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src import etl


def get_run_date():
    context = get_current_context()
    # In the UI, optionally pass a specific day through "run_date".
    chosen = context["params"]["run_date"]
    return chosen or context["data_interval_start"].strftime("%Y-%m-%d")


@dag(
    dag_id="retail_data_pipeline",
    start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    schedule="@daily",
    catchup=False,  # Backfills are shown manually in the class.
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=10)},
    params={"run_date": "", "demo_retry": False},
    tags=["workshop", "data-engineering"],
)
def data_pipeline():
    @task
    def extract_to_bronze():
        context = get_current_context()
        if context["params"]["demo_retry"]:
            # Fail ONLY the first attempt of this DAG run to teach retries.
            run_id = context["ti"].run_id
            marker_id = hashlib.sha256(run_id.encode()).hexdigest()[:12]
            marker = etl.DATA / "demo" / f"retry_{marker_id}.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            if not marker.exists():
                marker.write_text("first attempt failed", encoding="utf-8")
                raise RuntimeError("Demo failure: Airflow will retry in 10 seconds")
        etl.extract_to_bronze(get_run_date())

    @task
    def transform_to_silver():
        etl.transform_to_silver(get_run_date())

    @task
    def prepare_gold():
        etl.prepare_gold(get_run_date())

    @task
    def validate_gold():
        etl.validate_gold(get_run_date())

    @task
    def persist_gold_to_postgres():
        etl.load_gold_to_postgres(get_run_date())

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_ml_training",
        trigger_dag_id="retail_ml_training_pipeline",
    )

    bronze = extract_to_bronze()
    silver = transform_to_silver()
    gold = prepare_gold()
    validated = validate_gold()
    persisted = persist_gold_to_postgres()

    bronze >> silver >> gold >> validated >> persisted >> trigger_training


data_pipeline()
