"""
app.intelligence.ml.evaluation
=================================

Metric computation for both Task 6 problems. Every function here is a pure
function of ``(y_true, y_pred[, y_proba])`` — no model objects, no I/O — so
it can be unit tested and reused identically for CV-fold metrics and final
holdout metrics.

Metrics that are mathematically undefined for a particular fold (most
notably ROC-AUC/PR-AUC when a validation split contains only one class,
which happens routinely here because the synthetic pit schedule clusters
every pit event at laps 18/27/36) are reported as ``None`` with a reason
string rather than silently dropped or replaced by a fabricated number.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None

    mape = None
    if np.all(y_true != 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "n": int(len(y_true))}


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes_true = len(np.unique(y_true))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    roc_auc = None
    pr_auc = None
    undefined_reason = None
    if n_classes_true < 2:
        undefined_reason = (
            "Only one class present in this split's ground truth "
            "(synthetic pit schedule clusters pit events at specific laps); "
            "ROC-AUC and PR-AUC are mathematically undefined."
        )
    elif y_proba is not None:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [0, 1],
        "undefined_reason": undefined_reason,
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
    }


def aggregate_metrics(per_fold_metrics: list[dict], metric_keys: list[str]) -> dict:
    """Mean/std of each metric across folds, skipping ``None`` values and
    recording how many folds actually contributed to each aggregate."""
    summary: dict = {}
    for key in metric_keys:
        values = [m.get(key) for m in per_fold_metrics if m.get(key) is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_folds": len(values),
            }
        else:
            summary[key] = {"mean": None, "std": None, "n_folds": 0}
    summary["n_folds_total"] = len(per_fold_metrics)
    return summary
