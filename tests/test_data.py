"""Cleaning, validation and end-to-end smoke tests."""
import pandas as pd

from src.data.validation import DataValidationError, validate_transactions
from src.features.builder import build_customer_dataset
from tests.helpers import CFG, make_clean, small_clean


def test_exact_duplicates_removed_returns_flagged_not_dropped():
    df = make_clean([
        (1, "A1", "2011-06-01 10:00", "10001", 2, 5.0, "red mug "),
        (1, "A1", "2011-06-01 10:00", "10001", 2, 5.0, "red mug "),       # exact duplicate
        (1, "C9", "2011-06-02 10:00", "10001", -1, 5.0, "RED  MUG"),
        (1, "A3", "2011-06-03 10:00", "POST", 1, 18.0, "POSTAGE"),
        (None, "A4", "2011-06-03 10:00", "10001", 1, 5.0, "RED MUG"),
        (1, "A5", "2011-06-03 10:00", "10001", 1, 0.0, "RED MUG"),
    ])
    assert len(df) == 5
    assert df["is_return"].sum() == 1
    assert (~df["is_product"]).sum() == 1 and (~df["has_customer"]).sum() == 1 and (~df["valid_price"]).sum() == 1
    assert set(df["description"].dropna()) == {"RED MUG", "POSTAGE"}          # whitespace/case normalised


def test_validation_reports_all_failures():
    df = small_clean(n_customers=50)
    bad = df.copy()
    bad.loc[bad.index[:200], "customer_id"] = pd.NA
    bad["customer_id"] = bad["customer_id"].where(bad.index % 2 == 0)         # push null rate > 35%
    bad.loc[bad.index[0], "invoice_date"] = pd.Timestamp("2005-01-01")
    try:
        validate_transactions(bad)
    except DataValidationError as exc:
        msg = str(exc)
        assert "customer_id" in msg and "invoice_date range" in msg            # both problems reported
    else:
        raise AssertionError("expected DataValidationError")


def test_validation_passes_on_good_data():
    assert validate_transactions(small_clean(n_customers=50))["rows"] > 0


def test_pipeline_dataset_smoke():
    ds = build_customer_dataset(small_clean(n_customers=400), CFG)
    assert {"customer_id", "churned", "clv_target", "recency_days"} <= set(ds.columns)
    assert 0.1 < ds["churned"].mean() < 0.9
    assert (ds.loc[ds["churned"] == 1, "clv_target"] == 0).all()
