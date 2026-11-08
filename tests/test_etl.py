import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import etl


class ETLPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_dir = etl.DATA
        etl.DATA = Path(self.temp_dir.name) / "data"

    def tearDown(self):
        etl.DATA = self.original_data_dir
        self.temp_dir.cleanup()

    def test_file_for_rejects_invalid_date(self):
        with self.assertRaises(ValueError):
            etl.file_for("raw", "2026-02-30")

    def test_raw_to_gold_removes_invalid_and_duplicate_rows(self):
        run_date = "2026-10-01"
        raw_dir = etl.file_for("raw", run_date).parent
        raw_dir.mkdir(parents=True)
        rows = [
            {
                "order_id": f"order-{index}",
                "order_date": run_date,
                "order_hour": index % 24,
                "customer_region": "Cairo",
                "warehouse_id": "WH-01",
                "shipping_method": "Standard",
                "distance_km": float(index + 1),
                "items_count": index % 5 + 1,
                "package_weight_kg": float(index + 1),
                "order_value_egp": float(index + 100),
                "promised_delivery_days": 2,
                "warehouse_load_pct": 50.0,
                "weather_risk_level": "Low",
                "carrier_on_time_rate": 0.9,
                "late_delivery": index % 2,
            }
            for index in range(20)
        ]
        rows.extend(
            [
                rows[0].copy(),
                {**rows[1], "order_id": "missing-feature", "weather_risk_level": None},
                {**rows[2], "order_id": "zero-distance", "distance_km": 0},
                {**rows[3], "order_id": "nonpositive-value", "order_value_egp": 0},
            ]
        )
        pd.DataFrame(rows).to_csv(etl.file_for("raw", run_date), index=False)

        etl.extract_to_bronze(run_date)
        etl.transform_to_silver(run_date)
        etl.prepare_gold(run_date)
        etl.validate_gold(run_date)

        gold = pd.read_csv(etl.file_for("gold", run_date))
        self.assertEqual(len(gold), 20)
        self.assertTrue(gold["order_id"].is_unique)
        self.assertTrue(gold[etl.FEATURES + [etl.TARGET]].notna().all().all())
        self.assertEqual(
            list(gold.columns), ["order_id", "order_date"] + etl.FEATURES + [etl.TARGET]
        )


if __name__ == "__main__":
    unittest.main()