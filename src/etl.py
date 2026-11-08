import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"


FEATURES = [
    "order_hour",
    "customer_region",
    "warehouse_id",
    "shipping_method",
    "distance_km",
    "items_count",
    "package_weight_kg",
    "order_value_egp",
    "promised_delivery_days",
    "warehouse_load_pct",
    "weather_risk_level",
    "carrier_on_time_rate",
]

TARGET = "late_delivery"


def file_for(layer: str, run_date: str) -> Path:
    date.fromisoformat(run_date)  # Reject invalid dates before using the file path.
    return DATA / layer / f"orders_{run_date}.csv"


def extract_to_bronze(run_date: str):
    source = file_for("raw", run_date)
    target = file_for("bronze", run_date)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing source: {source}. Run generate_data.py to add dates."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)  # Bronze preserves the raw CSV exactly.
    print(f"Bronze: copied raw data for {run_date}")


def transform_to_silver(run_date: str):
    df = pd.read_csv(file_for("bronze", run_date))
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[(df["distance_km"] > 0) & (df["order_value_egp"] > 0)]
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.date.astype(str)
    target = file_for("silver", run_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)  # Overwrite this day's partition: idempotent.
    print(f"Silver: {before} raw rows -> {len(df)} clean rows")


def prepare_gold(run_date: str):
    df = pd.read_csv(file_for("silver", run_date))
    columns = ["order_id", "order_date"] + FEATURES + [TARGET]
    gold = df[columns].copy()
    target = file_for("gold", run_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(target, index=False)
    print(f"Gold: saved {len(gold)} rows with ML features and target")


def validate_gold(run_date: str):
    df = pd.read_csv(file_for("gold", run_date))
    assert len(df) >= 20, "Not enough clean orders"
    assert df["order_id"].is_unique, "Duplicate order IDs found"
    assert df[FEATURES + [TARGET]].notna().all().all(), "Missing values found"
    assert df[TARGET].isin([0, 1]).all(), "Unexpected target labels"
    print(f"Validation passed: {run_date}, {len(df)} orders")


def load_gold_to_postgres(run_date: str):
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    columns = ["order_id", "order_date"] + FEATURES + [TARGET]
    gold = pd.read_csv(file_for("gold", run_date))
    records = [
        tuple(value.item() if hasattr(value, "item") else value for value in row)
        for row in gold[columns].itertuples(index=False, name=None)
    ]
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS gold_orders (
            order_id TEXT PRIMARY KEY,
            order_date DATE NOT NULL,
            order_hour SMALLINT NOT NULL,
            customer_region TEXT NOT NULL,
            warehouse_id TEXT NOT NULL,
            shipping_method TEXT NOT NULL,
            distance_km DOUBLE PRECISION NOT NULL,
            items_count INTEGER NOT NULL,
            package_weight_kg DOUBLE PRECISION NOT NULL,
            order_value_egp DOUBLE PRECISION NOT NULL,
            promised_delivery_days SMALLINT NOT NULL,
            warehouse_load_pct DOUBLE PRECISION NOT NULL,
            weather_risk_level TEXT NOT NULL,
            carrier_on_time_rate DOUBLE PRECISION NOT NULL,
            late_delivery SMALLINT NOT NULL CHECK (late_delivery IN (0, 1))
        )
    """
    placeholders = ", ".join(["%s"] * len(columns))
    update_columns = [column for column in columns if column != "order_id"]
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in update_columns
    )
    upsert_sql = (
        f"INSERT INTO gold_orders ({', '.join(columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (order_id) DO UPDATE SET {updates}"
    )

    connection = PostgresHook(postgres_conn_id="delivery_postgres").get_conn()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                cursor.executemany(upsert_sql, records)
    finally:
        connection.close()

    print(f"PostgreSQL: upserted {len(records)} Gold orders for {run_date}")
