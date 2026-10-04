"""Load and standardize the raw Online Retail II workbook (PRD section 6.1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

COLUMN_MAP = {
    "Invoice": "invoice",
    "InvoiceNo": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "UnitPrice": "price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}

STANDARD_COLUMNS = [
    "invoice", "stock_code", "description", "quantity",
    "invoice_date", "price", "customer_id", "country",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to the project's snake_case schema."""
    out = df.rename(columns=COLUMN_MAP)
    missing = [c for c in STANDARD_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Raw data is missing expected columns: {missing}")
    return out[STANDARD_COLUMNS].copy()


def load_raw(path: str | Path, sheets: list[str] | None = None) -> pd.DataFrame:
    """Load every sheet of the workbook (or a CSV), combine and sort chronologically.

    Args:
        path: ``.xlsx`` workbook or ``.csv`` file.
        sheets: Sheet names to load for Excel input. ``None`` loads all sheets.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Online Retail II (see PRD section 5.5) "
            "or run the pipeline with --synthetic."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frames = pd.read_excel(path, sheet_name=sheets if sheets else None)
        raw = pd.concat(frames.values(), ignore_index=True)
    else:
        raw = pd.read_csv(path)
    df = standardize_columns(raw)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    log.info("Loaded %d raw rows from %s", len(df), path.name)
    return df
