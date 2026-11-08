# Delivery Delay: Airflow and Machine Learning

An end-to-end data pipeline that cleans daily order files, stores validated Gold
orders in PostgreSQL, and automatically trains a delivery-delay classifier.

## 1. Pull the project

Clone the repository and move into the project directory:

```powershell
git clone https://github.com/mahmoudrabea01/delivery_delay_airflow_ML_project.git
cd delivery_delay_airflow_ML_project
```

To update an existing checkout, run `git pull` from the project directory.

## 2. Run the project

### Prerequisites

- Git
- Docker Desktop with Docker Compose enabled

### Configure and start

Create the local environment file from the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace `POSTGRES_PASSWORD` with a local URL-safe password.
Then build and start Airflow and PostgreSQL:

```powershell
docker compose up --build -d
docker compose ps
```

Open the Airflow UI at [http://localhost:8080](http://localhost:8080). The
standalone Airflow service prints its login details in its logs:

```powershell
docker compose logs airflow
```

### Run the data pipeline

The data DAG is named `retail_data_pipeline`. To process a particular input
file, trigger this DAG from the Airflow UI and provide a `run_date` matching a
file under `data/raw`, for example:

```json
{"run_date": "2026-10-01"}
```

After the Gold data passes validation, the pipeline upserts it into PostgreSQL
and automatically triggers `retail_ml_training_pipeline`. The ML DAG requires
at least 50 Gold orders in PostgreSQL before it can train.

PostgreSQL is exposed locally on `localhost:5432`. Stop the services while
keeping the database volume with:

```powershell
docker compose down
```

### Run ETL tests

With Python, pandas, and NumPy installed, run:

```powershell
python -m unittest discover -s tests -v
```

## 3. Problems encountered and solutions

| Problem | Solution |
| --- | --- |
| Missing values in input rows | The Silver step drops rows missing required model features or the target. Gold validation checks that required fields are populated. |
| Duplicate orders | The Silver step removes duplicate `order_id` values. PostgreSQL also uses `order_id` as its primary key, so loading a repeated order updates the existing row. |
| Invalid distance or order value | The Silver step removes rows where `distance_km` or `order_value_egp` is not positive. |
| Identifier and date are not useful model inputs | `order_id` and `order_date` are retained in Gold/PostgreSQL for identification and ordering, but excluded from the model feature list. |
| Categorical and numeric features need different preprocessing | `weather_risk_level` is ordinal encoded as Low, Medium, High; region, shipping method, and warehouse are one-hot encoded; numeric features are standardized. |
| Imbalanced target classes | The train/test split is stratified to preserve the class distribution in both sets. No resampling or class weighting is currently applied, so class imbalance may still affect model performance. |
| Reprocessing a daily partition could create duplicate database rows | Gold records are upserted into `gold_orders` on the `order_id` primary key. |
| Training could run before usable Gold data exists | The training step checks for PostgreSQL Gold rows and requires at least 50 orders. Training is triggered only after Gold validation and the database load succeed. |

## 4. Additional information

### Pipeline flow

```text
data/raw/orders_YYYY-MM-DD.csv
		-> Bronze (raw copy)
		-> Silver (deduplicated and cleaned)
		-> Gold (validated model columns)
		-> PostgreSQL: gold_orders
		-> retail_ml_training_pipeline
		-> model evaluation and saved model
```

### Main components

- `dags/data_pipeline.py`: daily Bronze-to-Gold ETL, validation, PostgreSQL load,
	and ML DAG trigger.
- `dags/ml_training_pipeline.py`: feature preparation, training, evaluation,
	and model saving tasks.
- `src/etl.py`: CSV transformations, Gold validation, and PostgreSQL upsert.
- `src/train.py`: PostgreSQL feature loading, preprocessing, model training, and
	evaluation.
- `tests/test_etl.py`: focused tests for date validation and the raw-to-Gold
	cleaning path.

### Outputs and persistence

- PostgreSQL table: `gold_orders`.
- PostgreSQL data persists in the Docker volume `postgres-data`.
- Training data and test split: `data/ml/`.
- Candidate and final models: `models/candidate_model.joblib` and
	`models/late_delivery_model.joblib`.
- Evaluation metrics: `metrics/metrics.json` (accuracy, precision, recall, and
	F1 score).

The default daily schedule has `catchup=False`. To process historical files,
trigger the data DAG manually with the appropriate `run_date` parameter.