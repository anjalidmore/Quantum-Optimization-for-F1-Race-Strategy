"""
app.intelligence.ml.selection
================================

Rank trained models and pick a winner per task, transparently.

Regression: primary = CV MAE (lower is better), reported alongside CV RMSE,
CV R^2, test MAE and test R^2 so a reader can see the full picture rather
than trusting one number.

Classification: primary = CV ROC-AUC (higher is better, mean over folds
where it was computable — see ``evaluation.py``), with PR-AUC and F1 as
secondary criteria and a hard rule against ranking by accuracy alone, since
pit events are rare (~3% positive).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelResult:
    model_name: str
    task: str
    cv_summary: dict
    test_metrics: dict
    fit_seconds: float
    best_params: dict
    status: str = "trained"  # "trained" | "skipped"
    skip_reason: str | None = None


def _cv_mean(cv_summary: dict, key: str) -> float | None:
    return cv_summary.get(key, {}).get("mean")


def rank_regression_models(results: list[ModelResult]) -> list[dict]:
    trained = [r for r in results if r.status == "trained"]
    ranked = sorted(trained, key=lambda r: (_cv_mean(r.cv_summary, "mae") is None, _cv_mean(r.cv_summary, "mae")))
    rows = []
    for i, r in enumerate(ranked):
        rows.append(
            {
                "rank": i + 1,
                "model": r.model_name,
                "cv_mae": _cv_mean(r.cv_summary, "mae"),
                "cv_rmse": _cv_mean(r.cv_summary, "rmse"),
                "cv_r2": _cv_mean(r.cv_summary, "r2"),
                "test_mae": r.test_metrics.get("mae"),
                "test_rmse": r.test_metrics.get("rmse"),
                "test_r2": r.test_metrics.get("r2"),
                "fit_seconds": r.fit_seconds,
                "selected": i == 0,
            }
        )
    skipped = [r for r in results if r.status == "skipped"]
    for r in skipped:
        rows.append({"rank": None, "model": r.model_name, "status": "skipped", "reason": r.skip_reason})
    return rows


def rank_classification_models(results: list[ModelResult]) -> list[dict]:
    trained = [r for r in results if r.status == "trained"]

    def sort_key(r: ModelResult):
        auc = _cv_mean(r.cv_summary, "roc_auc")
        pr = _cv_mean(r.cv_summary, "pr_auc")
        f1 = _cv_mean(r.cv_summary, "f1")
        # None sorts last regardless of direction.
        return (
            auc is None,
            -(auc or 0),
            pr is None,
            -(pr or 0),
            f1 is None,
            -(f1 or 0),
        )

    ranked = sorted(trained, key=sort_key)
    rows = []
    for i, r in enumerate(ranked):
        rows.append(
            {
                "rank": i + 1,
                "model": r.model_name,
                "cv_roc_auc": _cv_mean(r.cv_summary, "roc_auc"),
                "cv_pr_auc": _cv_mean(r.cv_summary, "pr_auc"),
                "cv_f1": _cv_mean(r.cv_summary, "f1"),
                "cv_precision": _cv_mean(r.cv_summary, "precision"),
                "cv_recall": _cv_mean(r.cv_summary, "recall"),
                "test_roc_auc": r.test_metrics.get("roc_auc"),
                "test_pr_auc": r.test_metrics.get("pr_auc"),
                "test_f1": r.test_metrics.get("f1"),
                "test_undefined_reason": r.test_metrics.get("undefined_reason"),
                "fit_seconds": r.fit_seconds,
                "selected": i == 0,
            }
        )
    skipped = [r for r in results if r.status == "skipped"]
    for r in skipped:
        rows.append({"rank": None, "model": r.model_name, "status": "skipped", "reason": r.skip_reason})
    return rows
