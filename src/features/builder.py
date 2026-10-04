"""Customer-level feature engineering with strict point-in-time discipline (PRD section 7).

LEAKAGE RULE: every feature is computed ONLY from rows with ``invoice_date < cutoff``.
Labels (churn, future spend) are computed ONLY from ``[cutoff, cutoff + window)``.
``build_features`` never sees future rows: the first thing it does is filter them out,
and the unit tests in ``tests/test_features.py`` verify that mutating or deleting
post-cutoff data leaves the features unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.cleaning import customer_view
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

SECONDS_PER_DAY = 86_400.0
LABEL_COLUMNS = ["churned", "clv_target"]
ID_COLUMNS = ["customer_id", "country"]


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / SECONDS_PER_DAY


def _is_purchase(df: pd.DataFrame) -> pd.Series:
    return (df["quantity"] > 0) & (~df["is_return"])


def assign_categories(descriptions: pd.Series, keywords: dict[str, list[str]]) -> pd.Series:
    """Map product descriptions to coarse categories by keyword (first listed category wins)."""
    cat = pd.Series("other", index=descriptions.index, dtype="object")
    desc = descriptions.astype("string")
    for name, kws in reversed(list(keywords.items())):  # reversed so earlier categories win
        pattern = "|".join(kws)
        cat[desc.str.contains(pattern, na=False, regex=True).to_numpy(dtype=bool)] = name
    return cat


def build_features(df: pd.DataFrame, cutoff: str | pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
    """Compute customer-level features as of ``cutoff`` (exclusive).

    Args:
        df: Cleaned transactions (output of ``clean_transactions``).
        cutoff: Prediction date. Only rows strictly before it are used.
        config: Project config (uses the ``features`` section).

    Returns:
        One row per customer with at least one purchase before ``cutoff``.
        Indexed by ``customer_id``.
    """
    fcfg = config["features"]
    cutoff = pd.Timestamp(cutoff)

    hist = customer_view(df[df["invoice_date"] < cutoff])  # <-- the leakage guard
    purch = hist[_is_purchase(hist)]
    rets = hist[hist["is_return"]]
    if purch.empty:
        raise ValueError("No purchases before the cutoff; cannot build features")

    # ---- order-level table ------------------------------------------------
    orders = (
        purch.groupby(["customer_id", "invoice"], as_index=False)
        .agg(order_date=("invoice_date", "min"), order_value=("line_total", "sum"),
             order_items=("quantity", "sum"))
        .sort_values(["customer_id", "order_date"], kind="stable")
    )
    g = orders.groupby("customer_id")

    feat = pd.DataFrame({
        "frequency": g.size(),
        "monetary_total": g["order_value"].sum(),
        "monetary_avg": g["order_value"].mean(),
        "avg_basket_size": g["order_items"].mean(),
    })
    last_order, first_order = g["order_date"].max(), g["order_date"].min()
    feat["recency_days"] = _days(cutoff - last_order)
    feat["tenure_days"] = _days(cutoff - first_order)

    # Variance of gaps between consecutive orders (0 when fewer than 2 gaps exist)
    orders["gap_days"] = _days(g["order_date"].diff())
    feat["gap_variance"] = orders.groupby("customer_id")["gap_days"].var(ddof=0).fillna(0.0)

    # ---- purchase-frequency trend: OLS slope of orders per 30-day bin -------
    n_bins, bin_days = int(fcfg["trend_lookback_bins"]), float(fcfg["trend_bin_days"])
    age_bin = np.floor(_days(cutoff - orders["order_date"]) / bin_days).astype(int)
    in_window = (age_bin < n_bins).to_numpy()
    rows = feat.index.get_indexer(orders["customer_id"])[in_window]
    t_idx = (n_bins - 1 - age_bin.to_numpy())[in_window]  # 0 = oldest bin, n_bins-1 = most recent
    counts = np.zeros((len(feat), n_bins))
    np.add.at(counts, (rows, t_idx), 1.0)
    t = np.arange(n_bins, dtype=float)
    t_c = t - t.mean()
    feat["frequency_trend"] = (counts - counts.mean(axis=1, keepdims=True)) @ t_c / (t_c @ t_c)

    # ---- seasonal concentration: Herfindahl index of calendar-month shares ---
    month_counts = (
        orders.assign(month=orders["order_date"].dt.month)
        .groupby(["customer_id", "month"]).size().rename("n").reset_index()
    )
    month_counts["share"] = month_counts["n"] / month_counts["customer_id"].map(feat["frequency"])
    feat["seasonal_concentration"] = (month_counts["share"] ** 2).groupby(month_counts["customer_id"]).sum()

    # ---- returns, historical CLV ---------------------------------------------
    return_value = rets.groupby("customer_id")["line_total"].sum().abs()
    feat["return_rate"] = (return_value.reindex(feat.index).fillna(0.0) / feat["monetary_total"]).clip(0, 1)
    feat["n_return_lines"] = rets.groupby("customer_id").size().reindex(feat.index).fillna(0).astype(float)
    net = hist[_is_purchase(hist) | hist["is_return"]].groupby("customer_id")["line_total"].sum()
    feat["clv_historical"] = net.reindex(feat.index)

    # ---- discount sensitivity (PROXY): the dataset has no discount field ------
    # Share of a customer's lines priced below `threshold` x the pre-cutoff median price of
    # that SKU. The reference median uses pre-cutoff rows only (no future leakage).
    ref_price = purch.groupby("stock_code")["price"].median()
    discounted = purch["price"] < fcfg["discount_threshold"] * purch["stock_code"].map(ref_price)
    feat["discount_sensitivity"] = discounted.groupby(purch["customer_id"]).mean().reindex(feat.index)

    # ---- product breadth and category affinity vector ---------------------------
    feat["n_unique_products"] = purch.groupby("customer_id")["stock_code"].nunique()
    kw = fcfg["category_keywords"]
    cats = assign_categories(purch["description"], kw)
    qty_by_cat = purch.assign(cat=cats).pivot_table(
        index="customer_id", columns="cat", values="quantity", aggfunc="sum", fill_value=0.0
    )
    qty_by_cat = qty_by_cat.reindex(columns=[*kw.keys(), "other"], fill_value=0.0)
    shares = qty_by_cat.div(qty_by_cat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = [f"cat_share_{c}" for c in shares.columns]
    feat = feat.join(shares)

    # ---- country (latest known) ---------------------------------------------------
    latest_country = purch.sort_values("invoice_date", kind="stable").groupby("customer_id")["country"].last()
    feat["country"] = latest_country.astype("object")
    feat["is_uk"] = (feat["country"] == "United Kingdom").astype(float)

    feat = add_engagement_score(feat)
    feat.index.name = "customer_id"
    return feat


def add_engagement_score(feat: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-100 RFM engagement score from percentile ranks.

    Ranks are relative to the scored population. For single-customer API scoring the
    reference distribution (the training population) must be persisted and reused.
    """
    r = 1.0 - feat["recency_days"].rank(pct=True)
    f = feat["frequency"].rank(pct=True)
    m = feat["monetary_total"].rank(pct=True)
    out = feat.copy()
    out["engagement_score"] = 100.0 * (r + f + m) / 3.0
    return out


def build_labels(
    df: pd.DataFrame,
    customers: pd.Index,
    cutoff: str | pd.Timestamp,
    window_days: int,
    clv_horizon_days: int | None = None,
) -> pd.DataFrame:
    """Build churn and future-spend labels from ``[cutoff, cutoff + window)``.

    churned = 1 if the customer placed NO purchase inside the window.
    clv_target = net purchase revenue inside the CLV horizon (0 for churned customers).

    Raises:
        ValueError: if the window extends beyond the observed data. An unobserved window
            would label every customer as churned and silently bias the target.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon = int(clv_horizon_days or window_days)
    window_end = cutoff + pd.Timedelta(days=window_days)
    horizon_end = cutoff + pd.Timedelta(days=horizon)
    observed_until = df["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    if max(window_end, horizon_end) > observed_until:
        raise ValueError(
            f"Label window ends {max(window_end, horizon_end).date()} but data only covers "
            f"until {observed_until.date()}. Choose an earlier cutoff."
        )

    fut = customer_view(df[(df["invoice_date"] >= cutoff) & (df["invoice_date"] < max(window_end, horizon_end))])
    fut = fut[_is_purchase(fut)]
    in_churn_window = fut[fut["invoice_date"] < window_end]
    active = pd.Index(in_churn_window["customer_id"].unique())
    in_clv_window = fut[fut["invoice_date"] < horizon_end]
    spend = in_clv_window.groupby("customer_id")["line_total"].sum()

    labels = pd.DataFrame(index=customers)
    labels["churned"] = (~customers.isin(active)).astype(int)
    labels["clv_target"] = spend.reindex(customers).fillna(0.0)
    labels.index.name = "customer_id"
    return labels


def build_customer_dataset(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Features at the configured cutoff joined with churn / CLV labels."""
    ccfg = config["churn"]
    cutoff = pd.Timestamp(ccfg["cutoff_date"])
    feats = build_features(df, cutoff, config)
    labels = build_labels(df, feats.index, cutoff, ccfg["window_days"], ccfg.get("clv_horizon_days"))
    out = feats.join(labels).reset_index()
    log.info("Customer dataset: %d customers, churn rate %.1f%%", len(out), 100 * out["churned"].mean())
    return out
