from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data" / "raw"
OUTPUT_FILE = ROOT / "data" / "data_dictionary.csv"

COLUMN_METADATA = {
    "order_id": {
        "description": "Unique identifier for an order.",
        "unit": "",
        "role": "identifier",
    },
    "order_date": {
        "description": "Date the order was placed.",
        "unit": "date (YYYY-MM-DD)",
        "role": "date",
    },
    "order_hour": {
        "description": "Hour of the day when the order was placed.",
        "unit": "hour (0-23)",
        "role": "feature",
    },
    "customer_region": {
        "description": "Customer's delivery region.",
        "unit": "",
        "role": "feature",
    },
    "warehouse_id": {
        "description": "Identifier of the warehouse fulfilling the order.",
        "unit": "",
        "role": "feature",
    },
    "shipping_method": {
        "description": "Shipping service selected for the order.",
        "unit": "",
        "role": "feature",
    },
    "distance_km": {
        "description": "Delivery distance for the order.",
        "unit": "kilometers",
        "role": "feature",
    },
    "items_count": {
        "description": "Number of items in the order.",
        "unit": "items",
        "role": "feature",
    },
    "package_weight_kg": {
        "description": "Total package weight.",
        "unit": "kilograms",
        "role": "feature",
    },
    "order_value_egp": {
        "description": "Monetary value of the order.",
        "unit": "EGP",
        "role": "feature",
    },
    "promised_delivery_days": {
        "description": "Promised delivery duration for the order.",
        "unit": "days",
        "role": "feature",
    },
    "warehouse_load_pct": {
        "description": "Warehouse load when the order is processed.",
        "unit": "percent",
        "role": "feature",
    },
    "weather_risk_level": {
        "description": "Categorical weather risk level for the delivery.",
        "unit": "",
        "role": "feature",
    },
    "carrier_on_time_rate": {
        "description": "Historical on-time delivery rate of the carrier.",
        "unit": "rate (0-1)",
        "role": "feature",
    },
    "late_delivery": {
        "description": "Whether the order was delivered late (1) or on time (0).",
        "unit": "binary (0/1)",
        "role": "target",
    },
}


def build_data_dictionary() -> pd.DataFrame:
    files = sorted(RAW_DATA_DIR.glob("orders_*.csv"))
    if not files:
        raise FileNotFoundError(f"No order CSV files found in {RAW_DATA_DIR}")

    data = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    rows = []

    for column in data.columns:
        values = data[column].dropna()
        metadata = COLUMN_METADATA.get(
            column,
            {
                "description": "No description is defined for this column.",
                "unit": "",
                "role": "additional",
            },
        )
        rows.append(
            {
                "column": column,
                **metadata,
                "pandas_dtype": str(data[column].dtype),
                "missing_count": int(data[column].isna().sum()),
                "distinct_count": int(values.nunique()),
                "example_values": ", ".join(
                    values.astype(str).drop_duplicates().head(3)
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    data_dictionary = build_data_dictionary()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary.to_csv(OUTPUT_FILE, index=False)
    print(f"Data dictionary saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
