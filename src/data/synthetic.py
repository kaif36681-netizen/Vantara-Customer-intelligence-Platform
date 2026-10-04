"""Synthetic generator that mimics Online Retail II's structure and quirks.

Purpose: smoke-test the whole pipeline (and the unit tests) without needing the real
dataset. It is NOT a substitute for the real data: metrics from it are meaningless.
Output uses the ORIGINAL raw column names so the loader/standardizer is exercised too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ADJ = ["RED", "BLUE", "VINTAGE", "PINK", "GREEN", "RETRO", "CHRISTMAS", "SMALL", "LARGE"]
_NOUN = ["HEART MUG", "TEA CUP", "CANDLE HOLDER", "GREETING CARD", "SHOPPER BAG", "TOY DOLL",
         "PHOTO FRAME", "CAKE TIN", "NOTEBOOK", "PARTY BUNTING", "LANTERN", "CUSHION COVER",
         "GIFT WRAP", "KEYRING", "LUNCH BOX", "WALL CLOCK"]


def generate_synthetic_transactions(
    n_customers: int = 1500,
    start: str = "2009-12-01",
    end: str = "2011-12-09",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate raw-format transactions with a real (learnable) churn signal."""
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span = (t1 - t0).days

    n_items = 300
    codes = [f"{10000 + i}" for i in range(n_items)]
    descs = [f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}" for _ in range(n_items)]
    base_price = np.round(rng.lognormal(mean=1.0, sigma=0.7, size=n_items), 2).clip(0.3, 60)

    rows: list[tuple] = []
    inv_no = 500000

    def add_order(cust, when, country, make_return=False):
        nonlocal inv_no
        inv_no += 1
        n_lines = 1 + rng.poisson(4)
        items = rng.integers(0, n_items, size=n_lines)
        for i in items:
            qty = int(1 + rng.geometric(0.25))
            price = float(base_price[i]) * (0.8 if rng.random() < 0.1 else 1.0)
            rows.append((str(inv_no), codes[i], descs[i], qty, when, round(price, 2), cust, country))
        if rng.random() < 0.3:
            rows.append((str(inv_no), "POST", "POSTAGE", 1, when, 18.0, cust, country))
        if make_return:
            inv_no += 1
            i = items[0]
            rows.append((f"C{inv_no}", codes[i], descs[i], -int(1 + rng.integers(0, 3)),
                         when + pd.Timedelta(days=int(rng.integers(1, 10))),
                         round(float(base_price[i]), 2), cust, country))

    for k in range(n_customers):
        cust = float(12346 + k)
        country = "United Kingdom" if rng.random() < 0.9 else str(rng.choice(["France", "Germany", "EIRE"]))
        rate = float(rng.lognormal(np.log(1 / 45), 0.7))          # orders per day
        first = t0 + pd.Timedelta(days=int(rng.integers(0, int(span * 0.75))))
        lifetime = float(rng.exponential(700))                     # days active before churning
        stop = min(first + pd.Timedelta(days=lifetime), t1)
        ret_p = float(rng.beta(1, 25))
        n_orders = max(1, rng.poisson(rate * max((stop - first).days, 1)))
        times = first + pd.to_timedelta(np.sort(rng.uniform(0, max((stop - first).days, 1), n_orders)), unit="D")
        for when in times:
            when = when.floor("min") + pd.Timedelta(hours=int(rng.integers(8, 18)))
            add_order(cust, when, country, make_return=rng.random() < ret_p)

    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])

    # Guest orders with missing Customer ID (~22% of rows, like the real data)
    n_guest = int(len(df) * 0.28)
    guest = df.sample(n_guest, random_state=seed).copy()
    guest["Customer ID"] = np.nan
    guest["Invoice"] = guest["Invoice"].astype(str).radd("9")
    # Quirks: exact duplicates, zero-price adjustments, a few messy descriptions
    dupes = df.sample(int(len(df) * 0.01), random_state=seed + 1)
    zero_price = df.sample(30, random_state=seed + 2).assign(Price=0.0, **{"Customer ID": np.nan})
    df = pd.concat([df, guest, dupes, zero_price], ignore_index=True)
    messy = df.sample(frac=0.02, random_state=seed + 3).index
    df.loc[messy, "Description"] = df.loc[messy, "Description"].str.lower() + "  "

    df = df[(df["InvoiceDate"] >= t0) & (df["InvoiceDate"] <= t1 + pd.Timedelta(days=1))]
    return df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
