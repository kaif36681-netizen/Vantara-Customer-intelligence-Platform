"""Data cleaning (PRD sections 5.3 and 6.2).

Design rule: rows are FLAGGED rather than silently dropped wherever the information
is itself predictive (returns) or needed elsewhere (missing Customer ID rows remain
usable for product-level analysis). Only exact duplicates are removed.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def iqr_bounds(s: pd.Series, factor: float = 3.0) -> tuple[float, float]:
    """Return (lower, upper) IQR outlier bounds for a numeric series."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def clean_transactions(
    df: pd.DataFrame,
    non_product_codes: list[str] | None = None,
    non_product_prefixes: list[str] | None = None,
    drop_exact_duplicates: bool = True,
    iqr_factor: float = 3.0,
) -> pd.DataFrame:
    """Clean raw transactions and add analysis flags.

    Added columns:
        is_return: invoice starts with 'C' or quantity < 0 (kept, never dropped).
        has_customer: Customer ID present.
        is_product: stock code is a real product (not postage, fees, manual entries...).
        valid_price: price > 0 (zero/negative prices are adjustments).
        qty_outlier / price_outlier: IQR-based flags on purchase rows (flag only).
        line_total: quantity * price (negative for returns).

    Note on duplicates: only rows identical on EVERY column (invoice, SKU, quantity,
    price, timestamp, customer) are removed. A legitimately repeated line inside one
    order is indistinguishable from a duplicate at this granularity; toggle with
    ``drop_exact_duplicates`` if that trade-off is wrong for your data.
    """
    codes = {c.upper() for c in (non_product_codes or [])}
    prefixes = tuple(p.upper() for p in (non_product_prefixes or []))

    out = df.copy()
    out["invoice"] = out["invoice"].astype("string").str.strip()
    out["stock_code"] = _normalize_text(out["stock_code"])
    out["description"] = _normalize_text(out["description"])
    out["invoice_date"] = pd.to_datetime(out["invoice_date"])
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    out["country"] = out["country"].astype("string").str.strip()

    n_before = len(out)
    if drop_exact_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(out)

    # Canonical description per StockCode = most frequent cleaned description.
    modal_desc = (
        out.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iloc[0])
    )
    out["description"] = out["stock_code"].map(modal_desc).fillna(out["description"])

    starts_c = out["invoice"].str.startswith("C").fillna(False).astype(bool)
    out["is_return"] = starts_c | (out["quantity"] < 0)
    out["has_customer"] = out["customer_id"].notna()
    is_admin = out["stock_code"].isin(codes)
    if prefixes:
        is_admin |= out["stock_code"].str.startswith(prefixes).fillna(False).astype(bool)
    out["is_product"] = ~is_admin
    out["valid_price"] = out["price"] > 0
    out["line_total"] = out["quantity"] * out["price"]

    purchases = out[(~out["is_return"]) & out["valid_price"]]
    q_lo, q_hi = iqr_bounds(purchases["quantity"], iqr_factor)
    p_lo, p_hi = iqr_bounds(purchases["price"], iqr_factor)
    out["qty_outlier"] = (~out["is_return"]) & ((out["quantity"] < q_lo) | (out["quantity"] > q_hi))
    out["price_outlier"] = out["valid_price"] & ((out["price"] < p_lo) | (out["price"] > p_hi))

    log.info(
        "Cleaned: %d rows (dupes removed=%d, returns=%d, no_customer=%d, non_product=%d, bad_price=%d)",
        len(out), n_dupes, int(out["is_return"].sum()), int((~out["has_customer"]).sum()),
        int((~out["is_product"]).sum()), int((~out["valid_price"]).sum()),
    )
    return out


def customer_view(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for customer-level modeling: known customer, real product, valid price."""
    return df[df["has_customer"] & df["is_product"] & df["valid_price"]]
