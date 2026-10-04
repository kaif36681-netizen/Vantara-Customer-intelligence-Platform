"""Shared test helpers (plain functions - no pytest fixtures needed)."""
import pandas as pd

from src.data.cleaning import clean_transactions
from src.data.loader import standardize_columns
from src.data.synthetic import generate_synthetic_transactions
from src.utils.config import load_config

CFG = load_config()


def small_clean(n_customers: int = 300, seed: int = 7) -> pd.DataFrame:
    raw = standardize_columns(generate_synthetic_transactions(n_customers=n_customers, seed=seed))
    d = CFG["data"]
    return clean_transactions(raw, d["non_product_codes"], d["non_product_prefixes"])


def make_clean(rows: list[tuple]) -> pd.DataFrame:
    """rows: (customer, invoice, 'YYYY-MM-DD HH:MM', stock, qty, price, desc)."""
    raw = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "stock_code",
                                      "quantity", "price", "description"])
    raw["country"] = "United Kingdom"
    d = CFG["data"]
    return clean_transactions(raw[["invoice", "stock_code", "description", "quantity",
                                   "invoice_date", "price", "customer_id", "country"]],
                              d["non_product_codes"], d["non_product_prefixes"])
