# Customer Behavior Prediction Platform - Churn Prediction Core

Predicts which customers will **not** buy in the next 90 days, using the UCI
*Online Retail II* transactions (about 1M rows, Dec 2009 - Dec 2011).

## Project layout
```
config/config.yaml          every path, threshold and hyperparameter
src/data/loader.py          load xlsx/csv, rename columns
src/data/cleaning.py        duplicates, returns, non-product codes, outlier flags
src/data/validation.py      schema / null-rate / date-range checks (fails loudly)
src/data/synthetic.py       fake data in the same format (for testing without the dataset)
src/features/builder.py     point-in-time features + churn/CLV labels  <- the heart of the project
src/pipeline.py             raw -> clean -> validated -> feature table
src/models/train_baselines.py   LR, Decision Tree, KNN, Random Forest (+XGBoost/LightGBM if installed)
tests/                      13 tests incl. data-leakage tests
docs/VIVA_GUIDE.md          explanation + viva questions and answers
```

## Run it step by step

**1. Install Python 3.10+** and open a terminal in the project folder.

**2. Create a virtual environment and install packages**
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Quick check WITHOUT the dataset (about 30 seconds)**
```bash
python -m src.pipeline --synthetic
python -m src.models.train_baselines
```
This proves the whole code path works. The metrics are meaningless (fake data).

**4. Get the real data**
Download *Online Retail II* from https://archive.ics.uci.edu/dataset/502/online+retail+ii
and save it as `data/raw/online_retail_II.xlsx` (it has two sheets: "Year 2009-2010" and "Year 2010-2011").

**5. Build the cleaned data and the customer feature table**
```bash
python -m src.pipeline
```
Reading the xlsx takes a few minutes. Output: `data/interim/transactions_clean.csv` and
`data/processed/customer_features.csv` (one row per customer, features + `churned` + `clv_target`).

**6. Train and compare models**
```bash
python -m src.models.train_baselines
```
Output in `models_artifacts/`: `model_comparison.csv`, `churn_best.joblib`, `best_model.json`.

**7. Run the tests**
```bash
pytest -v
```

**Optional:** `pip install xgboost lightgbm` and re-run step 6 to add those two models.

## Troubleshooting
| Problem | Fix |
|---|---|
| `FileNotFoundError ... online_retail_II.xlsx` | Do step 4, or use `--synthetic` |
| `ModuleNotFoundError: src` | Run commands from the project root, using `python -m ...` |
| `DataValidationError` | Read the listed problems; adjust `config/config.yaml` thresholds only if justified |
| `ValueError: ... Choose an earlier cutoff` | `cutoff_date + window_days` goes past the last date in the data |
| Slow xlsx load | Open once in Excel, save as CSV, point `paths.raw` to the CSV |
