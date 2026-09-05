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
    REPO_ROOT,
    TASK5_FEATURES_CSV,
    ArtifactPaths,
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


def _run_regression(dataset, out: ArtifactPaths) -> dict:
    features, holdout, folds, X_dev, y_dev, X_test, y_test = _split(dataset, REGRESSION_TARGET)
    contract = dataset.contract

    results: list[ModelResult] = []
    per_model_artifacts: dict[str, dict] = {}
    out.models_laptime.mkdir(parents=True, exist_ok=True)

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

        artifact_path = out.models_laptime / f"{spec.name}.joblib"
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


def _run_classification(dataset, out: ArtifactPaths) -> dict:
    features, holdout, folds, X_dev, y_dev, X_test, y_test = _split(dataset, CLASSIFICATION_TARGET)
    contract = dataset.contract

    results: list[ModelResult] = []
    per_model_artifacts: dict[str, dict] = {}
    out.models_pit.mkdir(parents=True, exist_ok=True)

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

        artifact_path = out.models_pit / f"{spec.name}.joblib"
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


def _rel(path, root: Path | None = None) -> str:
    """Manifest paths, relative to the repository root.

    The frontend joins these onto the API's ``/artifacts`` mount, so they must
    never be absolute. When output is redirected outside the repo (the test
    suite's ``tmp_path``), repo-relative is undefined — fall back to a path
    relative to the artifact root, which keeps the value relative and keeps the
    "no absolute paths in the manifest" invariant true in both cases.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        base = (root or ArtifactPaths.default().root).resolve()
        try:
            return str(Path("artifacts") / resolved.relative_to(base))
        except ValueError:
            return resolved.name


def _rel_out(out: ArtifactPaths, path) -> str:
    return _rel(path, out.root)


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


def _generate_regression_figures(reg: dict, out: ArtifactPaths) -> list[str]:
    written = []
    written.append(_rel_out(out, viz.regression_model_comparison(reg["comparison"], out.figures / "regression_model_comparison.png")))

    best_name = reg["best_model"]
    if best_name:
        best = reg["artifacts"][best_name]
        y_test = reg["y_test"]
        y_pred = best["y_pred_test"]
        written.append(_rel_out(out, viz.prediction_vs_actual(y_test, y_pred, best_name, out.figures / "prediction_vs_actual.png")))
        written.append(_rel_out(out, viz.residual_distribution(y_test, y_pred, best_name, out.figures / "residuals.png")))
        written.append(
            _rel_out(out, viz.residuals_vs_predictions(y_test, y_pred, best_name, out.figures / "residuals_vs_predictions.png"))
        )
        if best["importance"]:
            written.append(
                _rel_out(out, viz.feature_importance_chart(best["importance"], best_name, out.figures / "feature_importance.png"))
            )
    return written


def _generate_classification_figures(clf: dict, out: ArtifactPaths) -> list[str]:
    written = []
    written.append(
        _rel_out(out, viz.classification_model_comparison(clf["comparison"], out.figures / "classification_model_comparison.png"))
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
        written.append(_rel_out(out, viz.roc_curves(roc_curves, out.figures / "roc_curves.png")))
        written.append(_rel_out(out, viz.precision_recall_curves(pr_curves, out.figures / "precision_recall_curves.png")))

    best_name = clf["best_model"]
    if best_name:
        best = clf["artifacts"][best_name]
        written.append(
            _rel(
                viz.confusion_matrix_plot(
                    best["test_metrics"]["confusion_matrix"], best_name, out.figures / "confusion_matrix.png"
                )
            )
        )
        if best["importance"]:
            written.append(
                _rel(
                    viz.feature_importance_chart(
                        best["importance"], best_name, out.figures / "classification_feature_importance.png"
                    )
                )
            )
        if best.get("y_proba_test") is not None:
            written.append(
                _rel(
                    viz.probability_distribution(
                        y_test, best["y_proba_test"], best_name, out.figures / "probability_distribution.png"
                    )
                )
            )
    return written


def _registry_entries(reg: dict, clf: dict, dataset, out: ArtifactPaths) -> list[ModelRegistryEntry]:
    entries = []
    for task_result, models_dir, xgb_status_fn in (
        (reg, out.models_laptime, reg_mod),
        (clf, out.models_pit, clf_mod),
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


def train_all(output_root: Path | None = None) -> dict:
    """Run the full Task 6 pipeline and write every artifact. Returns a
    JSON-safe summary dict (also used directly by the test suite).

    ``output_root`` redirects every write - models, metrics, figures, reports,
    registry and manifest - beneath one directory. It defaults to the committed
    ``artifacts/`` layout, so production behaviour is unchanged; the test suite
    passes a ``tmp_path`` so running the tests no longer rewrites tracked files.
    """
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    out.ensure()
    ensure_dirs()
    dataset = load_and_validate()

    reg = _run_regression(dataset, out)
    clf = _run_classification(dataset, out)

    _write_json(
        {
            "target": reg["target"],
            "features": reg["features"],
            "holdout": reg["holdout"],
            "models": {name: {"cv_summary": r.cv_summary, "test_metrics": r.test_metrics, "best_params": r.best_params, "status": r.status} for name, r in zip((r.model_name for r in reg["results"]), reg["results"])},
            "comparison": reg["comparison"],
            "best_model": reg["best_model"],
        },
        out.metrics / "regression_metrics.json",
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
        out.metrics / "classification_metrics.json",
    )
    _write_json(
        {
            "regression": {name: art["fold_metrics"] for name, art in reg["artifacts"].items()},
            "classification": {name: art["fold_metrics"] for name, art in clf["artifacts"].items()},
        },
        out.metrics / "cv_fold_metrics.json",
    )
    _write_json(
        {"regression": reg["comparison"], "classification": clf["comparison"]},
        out.metrics / "model_comparison.json",
    )

    reg_figures = _generate_regression_figures(reg, out)
    clf_figures = _generate_classification_figures(clf, out)

    registry_entries = _registry_entries(reg, clf, dataset, out)
    write_registry(registry_entries, out.model_registry_json)

    report_paths = reports_mod.generate_all(dataset, reg, clf, out)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _rel(TASK5_FEATURES_CSV),
        "dataset_source": dataset.contract.dataset_source,
        "synthetic_data_warning": not dataset.contract.is_real_data,
        "best_regression_model": reg["best_model"],
        "best_classification_model": clf["best_model"],
        "models": [_rel_out(out, p) for p in sorted(out.models_laptime.glob("*.joblib"))] + [_rel_out(out, p) for p in sorted(out.models_pit.glob("*.joblib"))],
        "metrics": [_rel_out(out, p) for p in sorted(out.metrics.glob("*.json"))],
        "figures": reg_figures + clf_figures,
        "reports": [_rel_out(out, p) for p in report_paths],
    }
    _write_json(manifest, out.manifest_json)

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
