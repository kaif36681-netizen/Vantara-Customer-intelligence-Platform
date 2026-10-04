"""Leakage and correctness tests for the feature/label builder (PRD section 7.5)."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.builder import build_features, build_labels
from tests.helpers import CFG, make_clean, small_clean

CUTOFF = pd.Timestamp("2011-09-09")


def test_features_identical_when_future_rows_removed():
    df = small_clean()
    full = build_features(df, CUTOFF, CFG)
    truncated = build_features(df[df["invoice_date"] < CUTOFF], CUTOFF, CFG)
    assert_frame_equal(full, truncated)


def test_features_unchanged_when_future_rows_are_corrupted():
    df = small_clean()
    base = build_features(df, CUTOFF, CFG)
    mutated = df.copy()
    fut = mutated["invoice_date"] >= CUTOFF
    mutated.loc[fut, ["quantity", "price", "line_total"]] = mutated.loc[fut, ["quantity", "price", "line_total"]] * 100
    mutated.loc[fut, "stock_code"] = "99999"
    assert_frame_equal(base, build_features(mutated, CUTOFF, CFG))


def test_order_exactly_at_cutoff_is_excluded_from_features_but_counts_in_label():
    df = make_clean([
        (1, "A1", "2011-06-01 10:00", "10001", 2, 5.0, "RED MUG"),
        (1, "A2", "2011-09-09 00:00", "10001", 1, 5.0, "RED MUG"),   # on the cutoff
    ])
    feats = build_features(df, CUTOFF, CFG)
    assert feats.loc[1, "frequency"] == 1                              # strict < cutoff
    df2 = pd.concat([df, make_clean([(2, "B1", "2011-12-08 23:00", "10002", 1, 1.0, "X")])])
    labels = build_labels(df2, feats.index, CUTOFF, 90)
    assert labels.loc[1, "churned"] == 0                                # purchase at t=cutoff is in-window


def test_customer_first_seen_after_cutoff_is_not_in_features():
    df = make_clean([
        (1, "A1", "2011-06-01 10:00", "10001", 1, 5.0, "RED MUG"),
        (2, "B1", "2011-10-01 10:00", "10001", 1, 5.0, "RED MUG"),
        (3, "C1", "2011-12-08 10:00", "10001", 1, 5.0, "RED MUG"),
    ])
    assert list(build_features(df, CUTOFF, CFG).index) == [1]


def test_hand_computed_rfm_values():
    df = make_clean([
        (1, "A1", "2011-08-30 12:00", "10001", 2, 10.0, "RED MUG"),    # 20
        (1, "A2", "2011-09-04 12:00", "10002", 1, 30.0, "BLUE CUP"),   # 30
        (1, "A3", "2011-09-05 12:00", "10003", 3, 10.0, "PINK BAG"),   # 30  (last order)
    ])
    f = build_features(df, CUTOFF, CFG).loc[1]
    assert f["frequency"] == 3
    assert f["monetary_total"] == 80.0
    assert abs(f["monetary_avg"] - 80 / 3) < 1e-9
    assert abs(f["recency_days"] - 3.5) < 1e-9                         # 5 Sep 12:00 -> 9 Sep 00:00
    assert abs(f["tenure_days"] - 9.5) < 1e-9
    assert abs(f["avg_basket_size"] - 2.0) < 1e-9


def test_return_rate_and_net_clv():
    df = make_clean([
        (1, "A1", "2011-08-01 10:00", "10001", 10, 10.0, "RED MUG"),         # 100
        (1, "CA2", "2011-08-05 10:00", "10001", -2, 10.0, "RED MUG"),        # -20 return
    ])
    f = build_features(df, CUTOFF, CFG).loc[1]
    assert abs(f["return_rate"] - 0.2) < 1e-9
    assert abs(f["clv_historical"] - 80.0) < 1e-9
    assert f["frequency"] == 1                                              # returns are not orders


def test_churn_label_and_future_spend():
    df = make_clean([
        (1, "A1", "2011-06-01 10:00", "10001", 1, 5.0, "RED MUG"),
        (2, "B1", "2011-06-01 10:00", "10001", 1, 5.0, "RED MUG"),
        (2, "B2", "2011-10-15 10:00", "10001", 4, 5.0, "RED MUG"),          # buys in window
        (3, "D1", "2011-06-01 10:00", "10001", 1, 5.0, "RED MUG"),
        (3, "C2X", "2011-10-15 10:00", "10001", -1, 5.0, "RED MUG"),        # return only -> still churned
        (9, "Z1", "2011-12-08 10:00", "10001", 1, 5.0, "RED MUG"),          # pins the end of observed data
    ])
    feats = build_features(df, CUTOFF, CFG)
    lab = build_labels(df, feats.index, CUTOFF, 90)
    assert lab.loc[1, "churned"] == 1 and lab.loc[2, "churned"] == 0 and lab.loc[3, "churned"] == 1
    assert lab.loc[2, "clv_target"] == 20.0 and lab.loc[1, "clv_target"] == 0.0


def test_unobserved_label_window_raises():
    df = small_clean(n_customers=100)
    feats = build_features(df, CUTOFF, CFG)
    try:
        build_labels(df, feats.index, pd.Timestamp("2011-11-30"), 90)
    except ValueError as exc:
        assert "Choose an earlier cutoff" in str(exc)
    else:
        raise AssertionError("expected ValueError for a window beyond the observed data")


def test_feature_table_is_clean():
    f = build_features(small_clean(), CUTOFF, CFG)
    assert f.index.is_unique
    assert not f.drop(columns="country").isna().any().any()
    assert (f["recency_days"] >= 0).all() and (f["frequency"] >= 1).all()
    assert f["engagement_score"].between(0, 100).all()
    assert f["seasonal_concentration"].between(1 / 12 - 1e-9, 1 + 1e-9).all()
    cat_cols = [c for c in f.columns if c.startswith("cat_share_")]
    assert np.allclose(f[cat_cols].sum(axis=1), 1.0)
