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
    precision_recall_curve,
    roc_curve,
)

from app.intelligence.charts import Axis, Chart, Series, histogram, write as write_chart

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def _num(v):
    """A numpy scalar as a plain float, for the JSON sidecar."""
    return float(v)


def _save(fig, path: Path, chart: Chart | None = None) -> Path:
    """Save the PNG and, beside it, the numbers that drew it.

    Both come from this one call so they cannot disagree: the caller passes the
    same arrays it just handed to matplotlib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if chart is not None:
        write_chart(path.stem, chart, figure=path)
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
    return _save(fig, path, Chart(
        kind="bar", title="Lap-time regression, model comparison",
        caption="Cross-validated and held-out test error for every model that trained.",
        x=Axis("Model", kind="category"), y=Axis("MAE", unit="s"),
        series=[Series("CV MAE", names, cv_mae), Series("Test MAE", names, test_mae)],
    ))


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
    return _save(fig, path, Chart(
        kind="bar", title="Pit-decision classification, model comparison",
        caption="Cross-validated ROC-AUC and F1 for every model with a defined AUC.",
        x=Axis("Model", kind="category"), y=Axis("Score"),
        series=[Series("CV ROC-AUC", names, auc), Series("CV F1", names, f1)],
    ))


def prediction_vs_actual(y_true, y_pred, model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual lap time (s)")
    ax.set_ylabel("Predicted lap time (s)")
    ax.set_title(f"Predicted vs Actual — {model_name} (holdout test)")
    ax.legend()
    return _save(fig, path, Chart(
        kind="scatter", title=f"Predicted vs actual lap time, {model_name}",
        caption="Held-out chronological test laps. The diagonal is a perfect prediction.",
        x=Axis("Actual lap time", unit="s"), y=Axis("Predicted lap time", unit="s"),
        series=[Series(model_name, y_true, y_pred)],
        extra={"identity_line": [_num(lo), _num(hi)]},
    ))


def residual_distribution(y_true, y_pred, model_name: str, path: Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted), s")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {model_name} (holdout test)")
    mids, counts = histogram(residuals, bins=20)
    return _save(fig, path, Chart(
        kind="histogram", title=f"Residual distribution, {model_name}",
        caption="Actual minus predicted, held-out test laps. Centred on zero is unbiased.",
        x=Axis("Residual", unit="s"), y=Axis("Laps"),
        series=[Series("Count", mids, counts)], extra={"marker_x": 0},
    ))


def residuals_vs_predictions(y_true, y_pred, model_name: str, path: Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.5, s=18)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted lap time (s)")
    ax.set_ylabel("Residual (actual - predicted), s")
    ax.set_title(f"Residuals vs Predictions — {model_name} (holdout test)")
    return _save(fig, path, Chart(
        kind="scatter", title=f"Residuals vs predictions, {model_name}",
        caption="A pattern here means the model is wrong in a structured way, not just noisy.",
        x=Axis("Predicted lap time", unit="s"), y=Axis("Residual", unit="s"),
        series=[Series(model_name, y_pred, residuals)], extra={"marker_y": 0},
    ))


def feature_importance_chart(importance: dict, model_name: str, path: Path, top_n: int = 15) -> Path:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(names))))
    ax.barh(names, values, color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    return _save(fig, path, Chart(
        kind="bar-horizontal", title=f"Feature importance, {model_name}",
        caption=f"Top {len(names)} features by the model's own importance measure.",
        x=Axis("Importance"), y=Axis("Feature", kind="category"),
        series=[Series("Importance", names, values)],
    ))


def roc_curves(curves: dict[str, tuple], path: Path) -> Path:
    """``curves``: {model_name: (y_true, y_proba)} for models where AUC is defined."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    series = []
    for name, (y_true, y_proba) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
        # The same curve the display just drew, from the same two arrays.
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        series.append(Series(name, fpr, tpr))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_title("ROC Curves (holdout test, models with a defined AUC)")
    ax.legend(fontsize=8)
    return _save(fig, path, Chart(
        kind="line", title="ROC curves",
        caption="Held-out test laps, models with a defined AUC. The diagonal is chance.",
        x=Axis("False positive rate"), y=Axis("True positive rate"),
        series=series, extra={"identity_line": [0, 1]},
    ))


def precision_recall_curves(curves: dict[str, tuple], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    series = []
    for name, (y_true, y_proba) in curves.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        series.append(Series(name, recall, precision))
    ax.set_title("Precision-Recall Curves (holdout test, models with a defined AUC)")
    ax.legend(fontsize=8)
    return _save(fig, path, Chart(
        kind="line", title="Precision-recall curves",
        caption="Held-out test laps. With few positive laps this says more than ROC.",
        x=Axis("Recall"), y=Axis("Precision"), series=series,
    ))


def confusion_matrix_plot(cm: list[list[int]], model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["No pit", "Pit"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name} (holdout test)")
    return _save(fig, path, Chart(
        kind="confusion", title=f"Confusion matrix, {model_name}",
        caption="Held-out test laps at the tuned threshold.",
        x=Axis("Predicted", kind="category"), y=Axis("Actual", kind="category"),
        series=[], extra={"labels": ["No pit", "Pit"], "matrix": [[int(v) for v in row] for row in cm]},
    ))


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
    neg_mids, neg_counts = histogram(y_proba[y_true == 0], bins=20)
    series = [Series("Actual: no pit", neg_mids, neg_counts)]
    if np.any(y_true == 1):
        pos_mids, pos_counts = histogram(y_proba[y_true == 1], bins=20)
        series.append(Series("Actual: pit", pos_mids, pos_counts))
    return _save(fig, path, Chart(
        kind="histogram", title=f"Predicted pit probability, {model_name}",
        caption="Separation between the two actual classes is what the threshold has to exploit.",
        x=Axis("Predicted probability of pit"), y=Axis("Laps"), series=series,
    ))
