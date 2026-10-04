"""Automated data validation that fails loudly (PRD section 6.2)."""
from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = [
    "invoice", "stock_code", "description", "quantity", "invoice_date",
    "price", "customer_id", "country",
]
CORE_FIELDS = ["invoice", "stock_code", "invoice_date", "quantity", "price"]


class DataValidationError(ValueError):
    """Raised when the transaction table violates one or more validation rules."""


def validate_transactions(
    df: pd.DataFrame,
    max_null_rate_customer_id: float = 0.35,
    max_null_rate_core_fields: float = 0.001,
    expected_date_range: tuple[str, str] = ("2009-12-01", "2011-12-10"),
) -> dict[str, float]:
    """Run schema, null-rate and date-range checks.

    Raises:
        DataValidationError: listing EVERY failed check, not just the first.

    Returns:
        A small dict of summary statistics for logging.
    """
    failures: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if len(df) == 0:
        raise DataValidationError("Transaction table is empty")

    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        failures.append("invoice_date is not a datetime column")
    if not pd.api.types.is_numeric_dtype(df["quantity"]):
        failures.append("quantity is not numeric")
    if not pd.api.types.is_numeric_dtype(df["price"]):
        failures.append("price is not numeric")

    for col in CORE_FIELDS:
        rate = float(df[col].isna().mean())
        if rate > max_null_rate_core_fields:
            failures.append(f"{col}: null rate {rate:.4%} exceeds {max_null_rate_core_fields:.4%}")

    cust_null = float(df["customer_id"].isna().mean())
    if cust_null > max_null_rate_customer_id:
        failures.append(f"customer_id: null rate {cust_null:.2%} exceeds {max_null_rate_customer_id:.2%}")

    if pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        lo, hi = pd.Timestamp(expected_date_range[0]), pd.Timestamp(expected_date_range[1])
        dmin, dmax = df["invoice_date"].min(), df["invoice_date"].max()
        if dmin < lo or dmax > hi + pd.Timedelta(days=1):
            failures.append(f"invoice_date range {dmin.date()}..{dmax.date()} outside {lo.date()}..{hi.date()}")

    if (df["quantity"] == 0).any():
        failures.append("quantity contains zeros")

    if failures:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(failures))

    stats = {"rows": float(len(df)), "customer_id_null_rate": cust_null}
    log.info("Validation passed: %s", stats)
    return stats
