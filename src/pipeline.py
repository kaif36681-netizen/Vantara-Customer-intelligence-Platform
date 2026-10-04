"""Single-command pipeline: raw data -> cleaned transactions -> customer feature table.

Usage:
    python -m src.pipeline                 # real data in data/raw/
    python -m src.pipeline --synthetic     # smoke test without the real dataset
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from src.data.cleaning import clean_transactions
from src.data.loader import load_raw, standardize_columns
from src.data.synthetic import generate_synthetic_transactions
from src.data.validation import validate_transactions
from src.features.builder import build_customer_dataset
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("pipeline")


def run(config_path: str | None = None, synthetic: bool = False):
    """Run every pipeline stage and persist interim + processed tables."""
    cfg = load_config(config_path)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if synthetic:
        log.warning("Using SYNTHETIC data - metrics are for smoke-testing only")
        raw = standardize_columns(generate_synthetic_transactions(seed=cfg["seed"]))
        raw = raw.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    else:
        raw = load_raw(resolve_path(cfg["paths"]["raw"]), cfg["data"]["sheets"])

    dcfg, vcfg = cfg["data"], cfg["validation"]
    clean = clean_transactions(
        raw, dcfg["non_product_codes"], dcfg["non_product_prefixes"],
        dcfg["drop_exact_duplicates"], dcfg["iqr_factor"],
    )
    validate_transactions(
        clean, vcfg["max_null_rate_customer_id"], vcfg["max_null_rate_core_fields"],
        tuple(vcfg["expected_date_range"]),
    )

    interim = resolve_path(cfg["paths"]["interim"])
    interim.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(interim, index=False)

    dataset = build_customer_dataset(clean, cfg)
    processed = resolve_path(cfg["paths"]["processed"])
    processed.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed, index=False)
    log.info("Wrote %s and %s", interim.name, processed.name)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run(args.config, args.synthetic)
