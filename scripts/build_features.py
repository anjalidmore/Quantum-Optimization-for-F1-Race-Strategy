#!/usr/bin/env python3
"""
build_features.py
=================

Runs Task 5 (feature engineering and feature selection) and writes the contract
every later task reads:

    data/processed/f1_features_selected.csv
    data/processed/feature_metadata.json

The logic lives in ``app.intelligence.features.build``; this script is the
command-line entry point, and ``scripts/build_all.py`` calls the same function.
``docs/notebooks/task5_feature_engineering.ipynb`` walks through the same code
with the intermediate tables shown.

Usage
-----
    python scripts/build_features.py                 # rebuild the contract
    python scripts/build_features.py --check         # report only, write nothing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.paths import DATA_PROCESSED_DIR, FASTF1_LAPS_CLEAN_CSV, REPO_ROOT  # noqa: E402
from app.intelligence.features import build as feature_build  # noqa: E402

log = logging.getLogger("build_features")


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 5 — feature engineering and selection")
    parser.add_argument("--clean-csv", type=Path, default=FASTF1_LAPS_CLEAN_CSV,
                        help="Task 4's cleaned laps (default: data/processed/fastf1_laps_clean.csv)")
    parser.add_argument("--out-dir", type=Path, default=DATA_PROCESSED_DIR,
                        help="where to write the feature matrix and metadata")
    parser.add_argument("--check", action="store_true",
                        help="run the pipeline but write nothing (useful for verifying determinism)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    result = feature_build.build(args.clean_csv, args.out_dir, write=not args.check)
    meta = result["metadata"]
    export = result["export"]
    selected = meta["selected_features"]

    log.info("rows x columns          : %d x %d", *export.shape)
    log.info("regression features     : %d", len(selected["target_laptime"]))
    log.info("classification features : %d", len(selected["target_pit_next_lap"]))
    log.info("warm-up rows dropped    : %d (first usable lap %d)",
             meta["preprocessing_contract"]["warmup_rows_dropped"],
             meta["preprocessing_contract"]["first_usable_lap"])
    log.info("CV MAE / R2 (selected)  : %.4f s / %.4f",
             meta["validation_scores"]["regression_cv_MAE_s"],
             meta["validation_scores"]["regression_cv_R2"])
    log.info("CV ROC-AUC (pit)        : %.4f", meta["validation_scores"]["classification_cv_AUC"])
    if args.check:
        log.info("--check: nothing written")
    else:
        for path in (result["csv_path"], result["json_path"]):
            log.info("wrote %s", Path(path).resolve().relative_to(REPO_ROOT))
        # The Task 5 written deliverables: FE report, correlation matrix, importance report.
        for name, path in feature_build.write_reports(result).items():
            log.info("wrote %s -> %s", name, Path(path).resolve().relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
