"""
app.intelligence.ml.visualize
================================

Every figure required by Task 6, rendered strictly from real predictions,
residuals and metrics computed by ``pipeline.py`` — nothing here accepts a
hard-coded number. If a model was skipped (e.g. XGBoost unavailable) it is
simply absent from the corresponding chart, not filled in with a placeholder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def regression_model_comparison(rows: list[dict], path: Path) -> Path:
    trained = [r for r in rows if r.get("cv_mae") is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["model"] for r in trained]
    cv_mae = [r["cv_mae"] for r in trained]
    test_mae = [r["test_mae"] for r in trained]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, cv_mae, width, label="CV MAE")
    ax.bar(x + width / 2, test_mae, width, label="Test MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Lap-Time Regression — Model Comparison")
    ax.legend()
    return _save(fig, path)


def classification_model_comparison(rows: list[dict], path: Path) -> Path:
    trained = [r for r in rows if r.get("cv_roc_auc") is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["model"] for r in trained]
    auc = [r["cv_roc_auc"] for r in trained]
    f1 = [r["cv_f1"] or 0 for r in trained]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, auc, width, label="CV ROC-AUC")
    ax.bar(x + width / 2, f1, width, label="CV F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pit-Decision Classification — Model Comparison")
    ax.legend()
    return _save(fig, path)


def prediction_vs_actual(y_true, y_pred, model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual lap time (s)")
    ax.set_ylabel("Predicted lap time (s)")
    ax.set_title(f"Predicted vs Actual — {model_name} (holdout test)")
    ax.legend()
    return _save(fig, path)


def residual_distribution(y_true, y_pred, model_name: str, path: Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted), s")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {model_name} (holdout test)")
    return _save(fig, path)


def residuals_vs_predictions(y_true, y_pred, model_name: str, path: Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.5, s=18)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted lap time (s)")
    ax.set_ylabel("Residual (actual - predicted), s")
    ax.set_title(f"Residuals vs Predictions — {model_name} (holdout test)")
    return _save(fig, path)


def feature_importance_chart(importance: dict, model_name: str, path: Path, top_n: int = 15) -> Path:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(names))))
    ax.barh(names, values, color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    return _save(fig, path)


def roc_curves(curves: dict[str, tuple], path: Path) -> Path:
    """``curves``: {model_name: (y_true, y_proba)} for models where AUC is defined."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, (y_true, y_proba) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_title("ROC Curves (holdout test, models with a defined AUC)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def precision_recall_curves(curves: dict[str, tuple], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, (y_true, y_proba) in curves.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves (holdout test, models with a defined AUC)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def confusion_matrix_plot(cm: list[list[int]], model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["No pit", "Pit"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name} (holdout test)")
    return _save(fig, path)


def probability_distribution(y_true, y_proba, model_name: str, path: Path) -> Path:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_proba[y_true == 0], bins=20, alpha=0.6, label="Actual: no pit", color="#4C72B0")
    if np.any(y_true == 1):
        ax.hist(y_proba[y_true == 1], bins=20, alpha=0.6, label="Actual: pit", color="#C44E52")
    ax.set_xlabel("Predicted probability of pit")
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted Probability Distribution — {model_name}")
    ax.legend()
    return _save(fig, path)
