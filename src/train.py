import json
import os
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.etl import FEATURES, TARGET

ROOT = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))


def prepare_features():
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(postgres_conn_id="delivery_postgres")
    df = hook.get_pandas_df("SELECT * FROM gold_orders ORDER BY order_date, order_id")
    if df.empty:
        raise ValueError(
            "No validated Gold rows in PostgreSQL yet. Run the Data DAG first!"
        )
    df = df.drop_duplicates(subset=["order_id"])
    if len(df) < 50:
        raise ValueError("Please process at least 50 Gold orders before training.")
    target = ROOT / "data" / "ml" / "training.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    print(f"ML data: {len(df)} orders loaded from PostgreSQL")


def train_model():
    df = pd.read_csv(ROOT / "data" / "ml" / "training.csv")
    ordinal_features = ["weather_risk_level"]
    one_hot_features = ["customer_region", "shipping_method", "warehouse_id"]
    numeric_features = [
        feature
        for feature in FEATURES
        if feature not in ordinal_features + one_hot_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.25,
        random_state=42,
        stratify=df[TARGET],
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Low", "Medium", "High"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_features,
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
                one_hot_features,
            ),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=300))
    model.fit(X_train, y_train)

    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ROOT / "models" / "candidate_model.joblib")

    test = X_test.copy()
    test[TARGET] = y_test
    test.to_csv(ROOT / "data" / "ml" / "test.csv", index=False)
    print(f"Model trained on {len(X_train)} rows; {len(X_test)} reserved for testing")


def evaluate_model():
    model = joblib.load(ROOT / "models" / "candidate_model.joblib")
    test = pd.read_csv(ROOT / "data" / "ml" / "test.csv")
    predictions = model.predict(test[FEATURES])
    actual = test[TARGET]
    metrics = {
        "accuracy": round(float(accuracy_score(actual, predictions)), 3),
        "precision": round(
            float(precision_score(actual, predictions, zero_division=0)), 3
        ),
        "recall": round(float(recall_score(actual, predictions, zero_division=0)), 3),
        "f1": round(float(f1_score(actual, predictions, zero_division=0)), 3),
    }
    (ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "metrics" / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Evaluation: {metrics}")


def save_model():
    source = ROOT / "models" / "candidate_model.joblib"
    target = ROOT / "models" / "late_delivery_model.joblib"
    shutil.copyfile(source, target)
    print(f"Model saved: {target}")
