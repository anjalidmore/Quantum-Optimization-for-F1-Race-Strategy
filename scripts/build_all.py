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
    python scripts/build_all.py --skip-dl        # skip Tasks 7-8 (deep learning + XAI)
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
    ARTIFACTS_DIR,
    DATA_ENGINEERING_ARTIFACTS_DIR,
    DATA_RAW_DIR,
    DL_METRICS_JSON,
    EXPERT_SYSTEM_ARTIFACTS_DIR,
    FASTF1_LAPS_CLEAN_CSV,
    KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR,
    ML_MODEL_REGISTRY_JSON,
    SEARCH_ARTIFACTS_DIR,
    TASK5_FEATURE_METADATA_JSON,
    TASK5_FEATURES_CSV,
    XAI_RESULTS_JSON,
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


@_stage("Deep learning (Task 7)")
def build_dl(force: bool) -> dict | None:
    from app.intelligence.dl import pipeline as dl_pipeline

    if not force and dl_pipeline.artifacts_exist():
        log.info("Already built (%s exists). Use --force to regenerate.", DL_METRICS_JSON.name)
        # Anything that rewrote the shared registry wholesale may have dropped
        # Task 7's rows. They are reconstructable from the committed artifacts,
        # so restore them rather than leaving the API advertising nothing.
        restored = dl_pipeline.restore_registry_entries()
        if restored:
            log.info("Restored %d Task 7 registry entries from existing artifacts.", restored)
        return None
    return dl_pipeline.train_all(force=force)


@_stage("Explainable AI (Task 8)")
def build_xai(force: bool) -> dict | None:
    from app.intelligence.xai import pipeline as xai_pipeline

    if not force and xai_pipeline.artifacts_exist():
        log.info("Already built (%s exists). Use --force to regenerate.", XAI_RESULTS_JSON.name)
        return None
    return xai_pipeline.run_all()


@_stage("Artifact consistency validation")
def validate_artifacts() -> None:
    from app.intelligence.ml.registry import load_registry

    registry = load_registry()
    if not registry:
        raise SystemExit("Task 6 registry missing after build.")
    for entry in registry["models"]:
        artifact = entry.get("artifact")
        if not artifact:
            continue
        # Task 6 records paths relative to the repo root; Task 7 records them
        # relative to artifacts/. Resolve both rather than assuming one.
        if not ((_REPO_ROOT / artifact).exists() or (ARTIFACTS_DIR / artifact).exists()):
            raise SystemExit(f"Registry references missing artifact: {artifact}")
    n_deep = sum(1 for m in registry["models"] if m.get("family") == "deep")
    log.info("All %d registered model artifacts present on disk (%d classical, %d deep).",
             len(registry["models"]), len(registry["models"]) - n_deep, n_deep)


def _print_summary(ml_summary: dict | None, dl_summary: dict | None = None) -> None:
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
    if dl_summary:
        log.info("Task 7 (Deep Learning):")
        for target, r in dl_summary.items():
            log.info("  %s: %s", target, r["verdict"].replace("**", "").split(".")[0] + ".")
    elif DL_METRICS_JSON.exists():
        log.info("Task 7 (Deep Learning):   using existing artifacts (%s)", DL_METRICS_JSON.name)
    if XAI_RESULTS_JSON.exists():
        log.info("Task 8 (Explainable AI):  %s", XAI_RESULTS_JSON.name)
    log.info("=" * 70)
    log.info("Run the API:      uvicorn app.api.main:app --reload")
    log.info("Run the frontend: cd frontend && npm run dev")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the whole F1 Race Strategy Intelligence platform.")
    parser.add_argument("--force", action="store_true", help="Regenerate every stage even if artifacts already exist.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip Task 6 model training (the slowest stage).")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip Task 7 (deep learning) and Task 8 (explainable AI).")
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
    else:
        log.info("Skipping Task 6 (--skip-ml).")

    dl_summary = None
    if args.skip_dl:
        log.info("Skipping Tasks 7-8 (--skip-dl).")
    elif args.skip_ml and not ML_MODEL_REGISTRY_JSON.exists():
        # Task 7 compares against Task 6's committed results and Task 8
        # explains Task 6's persisted pipelines, so neither can run before it.
        log.warning("Skipping Tasks 7-8: they depend on Task 6, which was skipped and has "
                    "no existing artifacts.")
    else:
        dl_summary = build_dl(args.force)
        build_xai(args.force)

    if not args.skip_ml:
        validate_artifacts()

    _print_summary(ml_summary, dl_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
