"""
app.intelligence.ml.pipeline
===============================

Task 6 orchestrator. ``train_all()`` runs the complete, real pipeline end to
end:

    load + validate Task 5 contract
      -> chronological holdout + expanding-window CV folds (per task)
      -> bounded hyperparameter search for every configured model
      -> refit the winner on the full development set
      -> evaluate once on the untouched holdout test set
      -> persist pipelines, metrics, figures, reports, registry

Nothing here is mocked: every metric, figure and registry entry is derived
from an actual ``fit``/``predict`` call. If a step cannot run (e.g. XGBoost
unavailable), that is recorded as a skipped model with a reason — never
backfilled with a fabricated number.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.paths import (
    ARTIFACT_MANIFEST_JSON,
    ML_FIGURES_DIR,
    ML_METRICS_DIR,
    ML_MODELS_LAPTIME_DIR,
    ML_MODELS_PIT_DIR,
    REPO_ROOT,
    TASK5_FEATURES_CSV,
    ensure_dirs,
)
from app.intelligence.ml import classification as clf_mod
from app.intelligence.ml import regression as reg_mod
from app.intelligence.ml import reports as reports_mod
from app.intelligence.ml import visualize as viz
from app.intelligence.ml.data_contract import build_task_frame, get_xy, load_and_validate
from app.intelligence.ml.evaluation import aggregate_metrics, classification_metrics, regression_metrics
from app.intelligence.ml.persistence import save_pipeline
from app.intelligence.ml.registry import ModelRegistryEntry, write_registry
from app.intelligence.ml.selection import ModelResult, rank_classification_models, rank_regression_models
from app.intelligence.ml.splits import chronological_holdout, expanding_window_folds
from app.intelligence.ml.tuning import run_expanding_window_search

log = logging.getLogger("ml.pipeline")

REGRESSION_TARGET = "target_laptime"
CLASSIFICATION_TARGET = "target_pit_next_lap"
N_CV_FOLDS = 4
TEST_FRACTION = 0.2

REGRESSION_METRIC_KEYS = ["mae", "rmse", "r2", "mape"]
CLASSIFICATION_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def _split(dataset, target: str):
    frame, features = build_task_frame(dataset, target)
    holdout = chronological_holdout(frame, test_fraction=TEST_FRACTION)

    dev_df = frame.loc[holdout.dev_index].reset_index(drop=True)
    test_df = frame.loc[holdout.test_index].reset_index(drop=True)

    X_dev, y_dev = get_xy(dev_df, target, features)
    X_test, y_test = get_xy(test_df, target, features)

    folds = expanding_window_folds(dev_df, n_folds=N_CV_FOLDS)
    return features, holdout, folds, X_dev, y_dev, X_test, y_test


def _run_regression(dataset) -> dict:
    features, holdout, folds, X_dev, y_dev, X_test, y_test = _split(dataset, REGRESSION_TARGET)
    contract = dataset.contract

    results: list[ModelResult] = []
    per_model_artifacts: dict[str, dict] = {}
    ML_MODELS_LAPTIME_DIR.mkdir(parents=True, exist_ok=True)

    for spec in reg_mod.REGRESSION_MODEL_SPECS:
        log.info("Training regression model: %s", spec.name)
        build_pipeline = reg_mod.build_pipeline_factory(spec, features, contract)
        search = run_expanding_window_search(
            model_name=spec.name,
            task="laptime_regression",
            build_pipeline=build_pipeline,
            param_grid=spec.param_grid,
            X_dev=X_dev,
            y_dev=y_dev,
            folds=folds,
            metrics_fn=regression_metrics,
            metric_keys=REGRESSION_METRIC_KEYS,
            primary_metric="mae",
            higher_is_better=False,
            needs_proba=False,
            aggregate_fn=aggregate_metrics,
        )

        y_pred_test = search.final_pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, y_pred_test)

        artifact_path = ML_MODELS_LAPTIME_DIR / f"{spec.name}.joblib"
        save_pipeline(search.final_pipeline, artifact_path)

        importance = reg_mod.extract_feature_importance(spec, search.final_pipeline, features)

        results.append(
            ModelResult(
                model_name=spec.name,
                task="laptime_regression",
                cv_summary=search.best_cv_summary,
                test_metrics=test_metrics,
                fit_seconds=search.fit_seconds,
                best_params=search.best_params,
            )
        )
        per_model_artifacts[spec.name] = {
            "artifact_path": str(artifact_path),
            "fold_metrics": search.best_fold_metrics,
            "all_candidates": [
                {"params": c.params, "cv_summary": c.cv_summary} for c in search.all_candidates
            ],
            "test_metrics": test_metrics,
            "y_pred_test": y_pred_test,
            "importance": importance,
        }

    if not reg_mod.xgboost_available():
        results.append(
            ModelResult(
                model_name="xgboost",
                task="laptime_regression",
                cv_summary={},
                test_metrics={},
                fit_seconds=0.0,
                best_params={},
                status="skipped",
                skip_reason=f"XGBoost unavailable — skipped ({reg_mod.xgboost_unavailable_reason()})",
            )
        )

    comparison = rank_regression_models(results)
    best_row = next((r for r in comparison if r.get("selected")), None)

    return {
        "target": REGRESSION_TARGET,
        "features": features,
        "holdout": holdout.to_metadata(),
        "folds": [f.to_metadata() for f in folds],
        "results": results,
        "artifacts": per_model_artifacts,
        "comparison": comparison,
        "best_model": best_row["model"] if best_row else None,
        "X_test": X_test,
        "y_test": y_test,
    }


def _run_classification(dataset) -> dict:
    features, holdout, folds, X_dev, y_dev, X_test, y_test = _split(dataset, CLASSIFICATION_TARGET)
    contract = dataset.contract

    results: list[ModelResult] = []
    per_model_artifacts: dict[str, dict] = {}
    ML_MODELS_PIT_DIR.mkdir(parents=True, exist_ok=True)

    for spec in clf_mod.CLASSIFICATION_MODEL_SPECS:
        log.info("Training classification model: %s", spec.name)
        build_pipeline = clf_mod.build_pipeline_factory(spec, features, contract)
        search = run_expanding_window_search(
            model_name=spec.name,
            task="pit_decision_classification",
            build_pipeline=build_pipeline,
            param_grid=spec.param_grid,
            X_dev=X_dev,
            y_dev=y_dev,
            folds=folds,
            metrics_fn=classification_metrics,
            metric_keys=CLASSIFICATION_METRIC_KEYS,
            primary_metric="roc_auc",
            higher_is_better=True,
            needs_proba=True,
            aggregate_fn=aggregate_metrics,
        )

        y_pred_test = search.final_pipeline.predict(X_test)
        try:
            y_proba_test = search.final_pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba_test = None
        test_metrics = classification_metrics(y_test, y_pred_test, y_proba_test)

        artifact_path = ML_MODELS_PIT_DIR / f"{spec.name}.joblib"
        save_pipeline(search.final_pipeline, artifact_path)

        importance = clf_mod.extract_feature_importance(spec, search.final_pipeline, features)

        results.append(
            ModelResult(
                model_name=spec.name,
                task="pit_decision_classification",
                cv_summary=search.best_cv_summary,
                test_metrics=test_metrics,
                fit_seconds=search.fit_seconds,
                best_params=search.best_params,
            )
        )
        per_model_artifacts[spec.name] = {
            "artifact_path": str(artifact_path),
            "fold_metrics": search.best_fold_metrics,
            "all_candidates": [
                {"params": c.params, "cv_summary": c.cv_summary} for c in search.all_candidates
            ],
            "test_metrics": test_metrics,
            "y_pred_test": y_pred_test,
            "y_proba_test": y_proba_test,
            "importance": importance,
        }

    if not clf_mod.xgboost_available():
        results.append(
            ModelResult(
                model_name="xgboost",
                task="pit_decision_classification",
                cv_summary={},
                test_metrics={},
                fit_seconds=0.0,
                best_params={},
                status="skipped",
                skip_reason=f"XGBoost unavailable — skipped ({clf_mod.xgboost_unavailable_reason()})",
            )
        )

    comparison = rank_classification_models(results)
    best_row = next((r for r in comparison if r.get("selected")), None)

    return {
        "target": CLASSIFICATION_TARGET,
        "features": features,
        "holdout": holdout.to_metadata(),
        "folds": [f.to_metadata() for f in folds],
        "results": results,
        "artifacts": per_model_artifacts,
        "comparison": comparison,
        "best_model": best_row["model"] if best_row else None,
        "X_test": X_test,
        "y_test": y_test,
    }


def _rel(path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.Series):
        return _json_safe(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _generate_regression_figures(reg: dict) -> list[str]:
    written = []
    written.append(_rel(viz.regression_model_comparison(reg["comparison"], ML_FIGURES_DIR / "regression_model_comparison.png")))

    best_name = reg["best_model"]
    if best_name:
        best = reg["artifacts"][best_name]
        y_test = reg["y_test"]
        y_pred = best["y_pred_test"]
        written.append(_rel(viz.prediction_vs_actual(y_test, y_pred, best_name, ML_FIGURES_DIR / "prediction_vs_actual.png")))
        written.append(_rel(viz.residual_distribution(y_test, y_pred, best_name, ML_FIGURES_DIR / "residuals.png")))
        written.append(
            _rel(viz.residuals_vs_predictions(y_test, y_pred, best_name, ML_FIGURES_DIR / "residuals_vs_predictions.png"))
        )
        if best["importance"]:
            written.append(
                _rel(viz.feature_importance_chart(best["importance"], best_name, ML_FIGURES_DIR / "feature_importance.png"))
            )
    return written


def _generate_classification_figures(clf: dict) -> list[str]:
    written = []
    written.append(
        _rel(viz.classification_model_comparison(clf["comparison"], ML_FIGURES_DIR / "classification_model_comparison.png"))
    )

    y_test = clf["y_test"]
    roc_curves = {}
    pr_curves = {}
    for name, art in clf["artifacts"].items():
        proba = art.get("y_proba_test")
        if proba is not None and len(np.unique(y_test)) >= 2:
            roc_curves[name] = (y_test, proba)
            pr_curves[name] = (y_test, proba)
    if roc_curves:
        written.append(_rel(viz.roc_curves(roc_curves, ML_FIGURES_DIR / "roc_curves.png")))
        written.append(_rel(viz.precision_recall_curves(pr_curves, ML_FIGURES_DIR / "precision_recall_curves.png")))

    best_name = clf["best_model"]
    if best_name:
        best = clf["artifacts"][best_name]
        written.append(
            _rel(
                viz.confusion_matrix_plot(
                    best["test_metrics"]["confusion_matrix"], best_name, ML_FIGURES_DIR / "confusion_matrix.png"
                )
            )
        )
        if best["importance"]:
            written.append(
                _rel(
                    viz.feature_importance_chart(
                        best["importance"], best_name, ML_FIGURES_DIR / "classification_feature_importance.png"
                    )
                )
            )
        if best.get("y_proba_test") is not None:
            written.append(
                _rel(
                    viz.probability_distribution(
                        y_test, best["y_proba_test"], best_name, ML_FIGURES_DIR / "probability_distribution.png"
                    )
                )
            )
    return written


def _registry_entries(reg: dict, clf: dict, dataset) -> list[ModelRegistryEntry]:
    entries = []
    for task_result, models_dir, xgb_status_fn in (
        (reg, ML_MODELS_LAPTIME_DIR, reg_mod),
        (clf, ML_MODELS_PIT_DIR, clf_mod),
    ):
        comparison_by_name = {row["model"]: row for row in task_result["comparison"]}
        for result in task_result["results"]:
            row = comparison_by_name.get(result.model_name, {})
            xgb_status = None
            if result.model_name == "xgboost":
                xgb_status = (
                    "available" if xgb_status_fn.xgboost_available() else f"unavailable: {xgb_status_fn.xgboost_unavailable_reason()}"
                )
            entries.append(
                ModelRegistryEntry(
                    model_name=result.model_name,
                    task=task_result["target"],
                    target=task_result["target"],
                    features=task_result["features"],
                    validation={
                        "strategy": "expanding_window_lap_forward",
                        "n_cv_folds": len(task_result["folds"]),
                        "holdout": task_result["holdout"],
                    },
                    metrics={"cv": result.cv_summary, "test": result.test_metrics},
                    artifact=_rel(models_dir / f"{result.model_name}.joblib") if result.status == "trained" else "",
                    hyperparameters=result.best_params,
                    random_state=42,
                    dataset=_rel(TASK5_FEATURES_CSV),
                    training_rows=task_result["holdout"]["dev_rows"],
                    test_rows=task_result["holdout"]["test_rows"],
                    cv_folds=len(task_result["folds"]),
                    is_selected_best=(row.get("selected", False)),
                    synthetic_data_warning=not dataset.contract.is_real_data,
                    xgboost_status=xgb_status,
                )
            )
    return entries


def train_all() -> dict:
    """Run the full Task 6 pipeline and write every artifact. Returns a
    JSON-safe summary dict (also used directly by the test suite)."""
    ensure_dirs()
    dataset = load_and_validate()

    reg = _run_regression(dataset)
    clf = _run_classification(dataset)

    _write_json(
        {
            "target": reg["target"],
            "features": reg["features"],
            "holdout": reg["holdout"],
            "models": {name: {"cv_summary": r.cv_summary, "test_metrics": r.test_metrics, "best_params": r.best_params, "status": r.status} for name, r in zip((r.model_name for r in reg["results"]), reg["results"])},
            "comparison": reg["comparison"],
            "best_model": reg["best_model"],
        },
        ML_METRICS_DIR / "regression_metrics.json",
    )
    _write_json(
        {
            "target": clf["target"],
            "features": clf["features"],
            "holdout": clf["holdout"],
            "models": {r.model_name: {"cv_summary": r.cv_summary, "test_metrics": r.test_metrics, "best_params": r.best_params, "status": r.status} for r in clf["results"]},
            "comparison": clf["comparison"],
            "best_model": clf["best_model"],
        },
        ML_METRICS_DIR / "classification_metrics.json",
    )
    _write_json(
        {
            "regression": {name: art["fold_metrics"] for name, art in reg["artifacts"].items()},
            "classification": {name: art["fold_metrics"] for name, art in clf["artifacts"].items()},
        },
        ML_METRICS_DIR / "cv_fold_metrics.json",
    )
    _write_json(
        {"regression": reg["comparison"], "classification": clf["comparison"]},
        ML_METRICS_DIR / "model_comparison.json",
    )

    reg_figures = _generate_regression_figures(reg)
    clf_figures = _generate_classification_figures(clf)

    registry_entries = _registry_entries(reg, clf, dataset)
    write_registry(registry_entries)

    report_paths = reports_mod.generate_all(dataset, reg, clf)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _rel(TASK5_FEATURES_CSV),
        "dataset_source": dataset.contract.dataset_source,
        "synthetic_data_warning": not dataset.contract.is_real_data,
        "best_regression_model": reg["best_model"],
        "best_classification_model": clf["best_model"],
        "models": [_rel(p) for p in sorted(ML_MODELS_LAPTIME_DIR.glob("*.joblib"))] + [_rel(p) for p in sorted(ML_MODELS_PIT_DIR.glob("*.joblib"))],
        "metrics": [_rel(p) for p in sorted(ML_METRICS_DIR.glob("*.json"))],
        "figures": reg_figures + clf_figures,
        "reports": [_rel(p) for p in report_paths],
    }
    _write_json(manifest, ARTIFACT_MANIFEST_JSON)

    log.info("Task 6 pipeline complete. Best regression model: %s | Best classification model: %s", reg["best_model"], clf["best_model"])

    return _json_safe(
        {
            "best_regression_model": reg["best_model"],
            "best_classification_model": clf["best_model"],
            "regression_comparison": reg["comparison"],
            "classification_comparison": clf["comparison"],
            "manifest": manifest,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    summary = train_all()
    print(json.dumps(summary, indent=2))
