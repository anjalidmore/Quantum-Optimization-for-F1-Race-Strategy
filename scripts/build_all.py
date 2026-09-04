#!/usr/bin/env python3
"""
build_all.py
=============

Top-level build for the whole F1 Race Strategy Intelligence platform. Runs
every computational-intelligence stage end to end and prints a final system
summary. Nothing here fabricates output: each stage either produces real
artifacts or the build fails loudly.

Usage
-----
    python scripts/build_all.py                 # build only what's missing
    python scripts/build_all.py --force          # regenerate everything
    python scripts/build_all.py --skip-ml        # skip Task 6 training (slow step)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from app.core.paths import (  # noqa: E402
    DATA_ENGINEERING_ARTIFACTS_DIR,
    DATA_RAW_DIR,
    EXPERT_SYSTEM_ARTIFACTS_DIR,
    FASTF1_LAPS_CLEAN_CSV,
    KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR,
    ML_MODEL_REGISTRY_JSON,
    SEARCH_ARTIFACTS_DIR,
    TASK5_FEATURE_METADATA_JSON,
    TASK5_FEATURES_CSV,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_all")


def _stage(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            log.info("=" * 70)
            log.info("STAGE: %s", name)
            log.info("=" * 70)
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            log.info("Stage '%s' finished in %.1fs", name, time.perf_counter() - t0)
            return result

        return wrapper

    return decorator


@_stage("Environment validation")
def validate_environment() -> bool:
    ok = True
    try:
        import fastapi  # noqa: F401
        import joblib  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        log.error("Missing required dependency: %s", exc)
        ok = False
    try:
        import xgboost  # noqa: F401

        log.info("XGBoost available.")
    except Exception as exc:
        log.warning("XGBoost unavailable — Task 6 will skip it (%s: %s)", type(exc).__name__, exc)
    return ok


@_stage("Knowledge representation (Task 1)")
def build_knowledge_base(force: bool) -> None:
    if not force and (KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR / "reports" / "validation_report.md").exists():
        log.info("Already built. Use --force to regenerate.")
        return
    from scripts.build_knowledge_base import main as run

    code = run(KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR)
    if code != 0:
        raise SystemExit("Task 1 build failed validation.")


@_stage("Expert system (Task 2)")
def build_expert_system(force: bool) -> None:
    if not force and (EXPERT_SYSTEM_ARTIFACTS_DIR / "reports" / "rule_validation_report.md").exists():
        log.info("Already built. Use --force to regenerate.")
        return
    from scripts.run_expert_system import main as run

    code = run(EXPERT_SYSTEM_ARTIFACTS_DIR)
    if code != 0:
        raise SystemExit("Task 2 rule-base validation failed.")


@_stage("State-space search (Task 3)")
def build_search(force: bool) -> None:
    if not force and (SEARCH_ARTIFACTS_DIR / "reports" / "comparison_report.md").exists():
        log.info("Already built. Use --force to regenerate.")
        return
    from scripts.run_search import main as run

    code = run(24, SEARCH_ARTIFACTS_DIR)
    if code != 0:
        raise SystemExit("Task 3 optimality invariant failed.")


@_stage("Data engineering & EDA (Task 4)")
def build_data_engineering(force: bool, regenerate_synthetic: bool) -> None:
    if not force and FASTF1_LAPS_CLEAN_CSV.exists():
        log.info("Already built (%s exists). Use --force to re-clean.", FASTF1_LAPS_CLEAN_CSV.name)
        return
    from scripts.run_eda import main as run

    # NOTE: `regenerate` here controls whether Task 4 overwrites data/raw/*.csv
    # with freshly generated *synthetic* data. It is intentionally NOT tied to
    # --force: --force re-runs cleaning/EDA on whatever is already in
    # data/raw/ (e.g. a real session fetched via fetch_real_session.py), so a
    # routine rebuild never silently clobbers real data with synthetic data.
    code = run(DATA_RAW_DIR, DATA_ENGINEERING_ARTIFACTS_DIR, regenerate=regenerate_synthetic)
    if code != 0:
        raise SystemExit("Task 4 data engineering failed.")


@_stage("Feature engineering (Task 5)")
def check_feature_engineering() -> None:
    if not (TASK5_FEATURES_CSV.exists() and TASK5_FEATURE_METADATA_JSON.exists()):
        raise SystemExit(
            "Task 5 outputs missing. Run docs/notebooks/task5_feature_engineering.ipynb "
            f"to produce {TASK5_FEATURES_CSV} and {TASK5_FEATURE_METADATA_JSON}."
        )
    log.info("Task 5 contract present: %s, %s", TASK5_FEATURES_CSV.name, TASK5_FEATURE_METADATA_JSON.name)


@_stage("Machine learning (Task 6)")
def build_ml(force: bool) -> dict | None:
    if not force and ML_MODEL_REGISTRY_JSON.exists():
        log.info("Already built (%s exists). Use --force to regenerate.", ML_MODEL_REGISTRY_JSON.name)
        return None
    from app.intelligence.ml.pipeline import train_all

    return train_all()


@_stage("Artifact consistency validation")
def validate_artifacts() -> None:
    from app.intelligence.ml.registry import load_registry

    registry = load_registry()
    if not registry:
        raise SystemExit("Task 6 registry missing after build.")
    for entry in registry["models"]:
        if entry["artifact"] and not (_REPO_ROOT / entry["artifact"]).exists():
            raise SystemExit(f"Registry references missing artifact: {entry['artifact']}")
    log.info("All %d registered model artifacts present on disk.", len(registry["models"]))


def _print_summary(ml_summary: dict | None) -> None:
    log.info("=" * 70)
    log.info("BUILD SUMMARY — F1 Race Strategy Intelligence Platform")
    log.info("=" * 70)
    log.info("Task 1 (Knowledge Representation): %s", KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR)
    log.info("Task 2 (Expert System):            %s", EXPERT_SYSTEM_ARTIFACTS_DIR)
    log.info("Task 3 (Search):                   %s", SEARCH_ARTIFACTS_DIR)
    log.info("Task 4 (Data Engineering):         %s", DATA_ENGINEERING_ARTIFACTS_DIR)
    log.info("Task 5 (Feature Engineering):      %s", TASK5_FEATURES_CSV)
    if ml_summary:
        log.info("Task 6 (Machine Learning):")
        log.info("  Best regression model:     %s", ml_summary["best_regression_model"])
        log.info("  Best classification model: %s", ml_summary["best_classification_model"])
    else:
        from app.intelligence.ml.registry import load_registry

        registry = load_registry()
        if registry:
            best_reg = next((m["model_name"] for m in registry["models"] if m["target"] == "target_laptime" and m["is_selected_best"]), None)
            best_clf = next((m["model_name"] for m in registry["models"] if m["target"] == "target_pit_next_lap" and m["is_selected_best"]), None)
            log.info("Task 6 (Machine Learning): using existing artifacts")
            log.info("  Best regression model:     %s", best_reg)
            log.info("  Best classification model: %s", best_clf)
    log.info("=" * 70)
    log.info("Run the API:      uvicorn app.api.main:app --reload")
    log.info("Run the frontend: cd frontend && npm run dev")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the whole F1 Race Strategy Intelligence platform.")
    parser.add_argument("--force", action="store_true", help="Regenerate every stage even if artifacts already exist.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip Task 6 model training (the slowest stage).")
    parser.add_argument(
        "--regenerate-synthetic",
        action="store_true",
        help="Overwrite data/raw/*.csv with freshly generated synthetic data. "
        "Never implied by --force, so a real session fetched via "
        "scripts/fetch_real_session.py is never silently clobbered.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if not validate_environment():
        return 1

    build_knowledge_base(args.force)
    build_expert_system(args.force)
    build_search(args.force)
    build_data_engineering(args.force, args.regenerate_synthetic)
    check_feature_engineering()

    ml_summary = None
    if not args.skip_ml:
        ml_summary = build_ml(args.force)
        validate_artifacts()
    else:
        log.info("Skipping Task 6 (--skip-ml).")

    _print_summary(ml_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
