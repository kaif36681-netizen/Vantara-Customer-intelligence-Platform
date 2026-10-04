"""Baseline churn models (PRD sections 8-9).

Usage:
    python -m src.models.train_baselines

Protocol: stratified 70/15/15 split; 5-fold CV ROC-AUC on TRAIN only; decision threshold
tuned on VALIDATION (highest precision subject to a recall target); TEST evaluated once at
the end. The "best" model is chosen by VALIDATION ROC-AUC, never by test results.
XGBoost / LightGBM are used automatically if installed; otherwise they are skipped.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features.builder import ID_COLUMNS, LABEL_COLUMNS
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric columns except identifiers and labels."""
    excluded = set(ID_COLUMNS) | set(LABEL_COLUMNS)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified train/val/test split (default 70/15/15)."""
    s, seed = cfg["split"], cfg["seed"]
    X, y = df[feature_columns(df)], df["churned"]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=s["test"], stratify=y, random_state=seed)
    val_frac = s["val"] / (s["train"] + s["val"])
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def get_models(cfg: dict, pos_weight: float) -> dict:
    """Model zoo. Class imbalance is handled with class weights (no resampling leakage risk)."""
    seed = cfg["seed"]
    cv = StratifiedKFold(cfg["modeling"]["cv_folds"], shuffle=True, random_state=seed)
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                class_weight="balanced", random_state=seed),
        "knn": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=25))]),
        "random_forest": RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [4, 6, 8, None],
             "min_samples_leaf": [1, 5, 10, 20], "max_features": ["sqrt", 0.5]},
            n_iter=cfg["modeling"]["rf_search_iterations"], cv=cv, scoring="roc_auc",
            random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          scale_pos_weight=pos_weight, eval_metric="auc",
                                          random_state=seed, n_jobs=-1)
    except ImportError:
        log.warning("xgboost not installed - skipping")
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                            class_weight="balanced", random_state=seed, verbose=-1)
    except ImportError:
        log.warning("lightgbm not installed - skipping")
    return models


def pick_threshold(y_val, p_val, min_recall: float) -> float:
    """Highest-precision threshold on the validation set with recall >= min_recall (else 0.5)."""
    precision, recall, thr = precision_recall_curve(y_val, p_val)
    ok = recall[:-1] >= min_recall
    if not ok.any():
        return 0.5
    best = np.argmax(np.where(ok, precision[:-1], -1.0))
    return float(thr[best])


def metrics(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    df = pd.read_csv(resolve_path(cfg["paths"]["processed"]))
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, cfg)
    log.info("Split sizes train/val/test = %d/%d/%d | churn rate train=%.2f", len(X_tr), len(X_va), len(X_te), y_tr.mean())
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv = StratifiedKFold(cfg["modeling"]["cv_folds"], shuffle=True, random_state=cfg["seed"])

    rows, fitted = [], {}
    for name, model in get_models(cfg, pos_weight).items():
        model.fit(X_tr, y_tr)
        if isinstance(model, RandomizedSearchCV):
            cv_auc, cv_std = float(model.best_score_), float("nan")
            model = model.best_estimator_
        else:
            scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
            cv_auc, cv_std = float(scores.mean()), float(scores.std())
        p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        thr = pick_threshold(y_va, p_va, cfg["modeling"]["min_recall_target"])
        row = {"model": name, "cv_auc": cv_auc, "cv_auc_std": cv_std, "val_auc": roc_auc_score(y_va, p_va),
               "threshold": thr, **{f"test_{k}": v for k, v in metrics(y_te, p_te, thr).items()}}
        rows.append(row)
        fitted[name] = model
        log.info("%-20s cv_auc=%.3f val_auc=%.3f test_auc=%.3f test_recall=%.3f", name, cv_auc,
                 row["val_auc"], row["test_roc_auc"], row["test_recall"])

    results = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    out_dir = resolve_path(cfg["paths"]["artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "model_comparison.csv", index=False)

    best = results.loc[0, "model"]  # chosen on VALIDATION AUC only
    joblib.dump({"model": fitted[best], "features": list(X_tr.columns), "threshold": float(results.loc[0, "threshold"])},
                out_dir / "churn_best.joblib")
    (out_dir / "best_model.json").write_text(json.dumps(
        {"best_model": best, "features": list(X_tr.columns), "threshold": float(results.loc[0, "threshold"])}, indent=2))
    log.info("Best model by validation AUC: %s (saved to %s)", best, out_dir)

    if hasattr(fitted[best], "feature_importances_"):
        imp = pd.Series(fitted[best].feature_importances_, index=X_tr.columns).sort_values(ascending=False)
        log.info("Top features:\n%s", imp.head(8).round(3).to_string())
    return results


if __name__ == "__main__":
    import sys
    run()
    if "--no-dashboard" not in sys.argv:
        from src.dashboard import build_dashboard
        build_dashboard()
