"""
app.intelligence.ml.selection
================================

Rank trained models and pick a winner per task, transparently.

Regression: primary = CV MAE (lower is better), reported alongside CV RMSE,
CV R^2, test MAE and test R^2 so a reader can see the full picture rather
than trusting one number.

**Generalisation guard.** Selecting purely on CV MAE once shipped a model with
a *negative* test R^2 - worse than predicting the training mean - while a
candidate with positive test R^2 sat unselected. CV and holdout disagreeing
that badly is a signal, not a detail: it means the CV winner does not transfer
to the held-out regime. ``rank_regression_models`` now refuses to select a
model whose test R^2 is negative when any candidate has a positive one, and
records *why* it overrode the CV ranking so the decision is auditable rather
than silent.

Classification: primary = **CV PR-AUC** (higher is better, mean over folds
where it was computable - see ``evaluation.py``). It used to be ROC-AUC, and
that was the wrong headline at this prevalence: with pit events at 4.8% of
laps, ROC-AUC is dominated by the majority class and stayed near 0.98 for
models that were useless as decision rules. PR-AUC asks the question that
actually matters - of the laps this model flags, how many are real pit windows?
ROC-AUC and F1 remain as reported secondary criteria, and ranking by accuracy
alone remains forbidden.
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
    # Classification only: the tuned decision threshold and how it was chosen.
    threshold: dict | None = None


def _cv_mean(cv_summary: dict, key: str) -> float | None:
    return cv_summary.get(key, {}).get("mean")


def rank_regression_models(results: list[ModelResult]) -> list[dict]:
    trained = [r for r in results if r.status == "trained"]
    ranked = sorted(trained, key=lambda r: (_cv_mean(r.cv_summary, "mae") is None, _cv_mean(r.cv_summary, "mae")))

    selected_index, guard = _apply_generalisation_guard(ranked)

    rows = []
    for i, r in enumerate(ranked):
        row = {
            "rank": i + 1,
            "model": r.model_name,
            "cv_mae": _cv_mean(r.cv_summary, "mae"),
            "cv_rmse": _cv_mean(r.cv_summary, "rmse"),
            "cv_r2": _cv_mean(r.cv_summary, "r2"),
            "test_mae": r.test_metrics.get("mae"),
            "test_rmse": r.test_metrics.get("rmse"),
            "test_r2": r.test_metrics.get("r2"),
            "fit_seconds": r.fit_seconds,
            "selected": i == selected_index,
        }
        if i == selected_index and guard:
            row["selection_warning"] = guard
        rows.append(row)
    skipped = [r for r in results if r.status == "skipped"]
    for r in skipped:
        rows.append({"rank": None, "model": r.model_name, "status": "skipped", "reason": r.skip_reason})
    return rows


def _apply_generalisation_guard(ranked: list[ModelResult]) -> tuple[int, str | None]:
    """Return the index to select, and a warning if the CV ranking was overridden.

    The CV winner is kept unless its holdout R^2 is negative *and* some other
    candidate's is positive. A negative R^2 means the model does worse than
    predicting the training mean on held-out data; shipping that as the
    platform's lap-time predictor while a model that generalises sits unselected
    is not defensible, however good the CV number is.

    Among positive-R^2 candidates the best CV MAE still wins, so the override is
    the narrowest possible: it changes the choice only when the CV winner is
    demonstrably broken out of sample.
    """
    if not ranked:
        return 0, None

    def test_r2(r: ModelResult):
        return r.test_metrics.get("r2")

    top = ranked[0]
    top_r2 = test_r2(top)
    if top_r2 is None or top_r2 >= 0:
        return 0, None

    # ranked is already ordered by CV MAE, so the first positive-R^2 entry is
    # the best CV performer among the models that generalise.
    for i, r in enumerate(ranked):
        r2 = test_r2(r)
        if r2 is not None and r2 >= 0:
            return i, (
                f"CV ranking overridden by the generalisation guard: the best CV MAE model "
                f"({top.model_name}, CV MAE {_cv_mean(top.cv_summary, 'mae'):.4f}) scores test "
                f"R2 {top_r2:.4f} — worse than predicting the mean. Selected {r.model_name} "
                f"instead (CV MAE {_cv_mean(r.cv_summary, 'mae'):.4f}, test R2 {r2:.4f}), the "
                f"best CV performer among candidates that generalise to the holdout. This "
                f"disagreement between CV and holdout is itself a finding: the holdout is the "
                f"closing laps of a single race, a different fuel and tyre regime from training."
            )

    return 0, (
        f"Every candidate scores a negative test R2 — the best CV model ({top.model_name}) is "
        f"selected by CV MAE, but no model in this run generalises to the holdout. Treat the "
        f"lap-time prediction as unvalidated out of sample."
    )


def rank_classification_models(results: list[ModelResult]) -> list[dict]:
    trained = [r for r in results if r.status == "trained"]

    def sort_key(r: ModelResult):
        pr = _cv_mean(r.cv_summary, "pr_auc")
        auc = _cv_mean(r.cv_summary, "roc_auc")
        f1 = _cv_mean(r.cv_summary, "f1")
        # PR-AUC leads: at a 4.8% positive rate ROC-AUC stays high for models
        # that never fire. None sorts last regardless of direction.
        return (
            pr is None,
            -(pr or 0),
            auc is None,
            -(auc or 0),
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
                "test_precision": r.test_metrics.get("precision"),
                "test_recall": r.test_metrics.get("recall"),
                "decision_threshold": (r.threshold or {}).get("threshold"),
                "threshold_objective": (r.threshold or {}).get("objective"),
                "test_undefined_reason": r.test_metrics.get("undefined_reason"),
                "fit_seconds": r.fit_seconds,
                "selected": i == 0,
            }
        )
    skipped = [r for r in results if r.status == "skipped"]
    for r in skipped:
        rows.append({"rank": None, "model": r.model_name, "status": "skipped", "reason": r.skip_reason})
    return rows
