"""
app.intelligence.dl.visualize
=============================

Training-history figures.

The accuracy/loss curves are the primary diagnostic deliverable for Task 7:
they are how a reader tells "still learning" from "plateaued" from
"overfitting". Each figure marks the early-stopping epoch explicitly, so the
effect of the overfitting countermeasures is visible rather than asserted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(history: dict, title: str, out_path: Path, best_epoch: int | None = None) -> Path:
    metric_key = "mae" if "mae" in history else ("auc" if "auc" in history else None)
    ncols = 2 if metric_key else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.2))
    axes = [axes] if ncols == 1 else list(axes)

    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label="training loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if metric_key:
        axes[1].plot(epochs, history[metric_key], label=f"training {metric_key}")
        vk = f"val_{metric_key}"
        if vk in history:
            axes[1].plot(epochs, history[vk], label=f"validation {metric_key}")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(metric_key.upper())
        axes[1].set_title(metric_key.upper())
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    if best_epoch:
        for ax in axes:
            ax.axvline(
                best_epoch, color="crimson", linestyle="--", linewidth=1,
                label=f"best epoch ({best_epoch})",
            )
        axes[0].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_model_comparison(rows: list[dict], metric: str, title: str, out_path: Path, lower_is_better: bool) -> Path:
    names = [r["model"] for r in rows]
    values = [r[metric] for r in rows]
    colors = ["#c0392b" if n.startswith("dnn") else "#7f8c8d" for n in names]

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(names) + 2.0))
    ax.barh(names, values, color=colors)
    ax.set_xlabel(f"{metric} ({'lower is better' if lower_is_better else 'higher is better'})")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Task 7 deliverable plots (one file per curve, as the specification lists them)
# ---------------------------------------------------------------------------
def plot_curve(history: dict, key: str, ylabel: str, title: str, out_path: Path,
               best_epoch: int | None = None) -> Path:
    """Training vs validation for one tracked quantity (``loss``, ``accuracy``,
    ``mae``). The dashed line marks the epoch whose weights early stopping
    restored — the model that was saved."""
    epochs = range(1, len(history[key]) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, history[key], label=f"training {key}")
    if f"val_{key}" in history:
        ax.plot(epochs, history[f"val_{key}"], label=f"validation {key}")
    if best_epoch:
        ax.axvline(best_epoch, color="crimson", ls="--", lw=1, label=f"best epoch ({best_epoch}) — restored")
    values = [v for k in (key, f"val_{key}") for v in history.get(k, []) if v and v > 0]
    if key == "loss" and values and max(values) / min(values) > 20:
        # A few huge early-epoch losses would otherwise flatten the part of the
        # curve that matters (the divergence after the best epoch).
        ax.set_yscale("log")
        ylabel = f"{ylabel} — log scale"
    ax.set(xlabel="epoch", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_confusion(cm, threshold: float, title: str, out_path: Path) -> Path:
    import numpy as np

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    labels = ["stay out (0)", "pit next lap (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="predicted", ylabel="actual", title=f"{title}\n(threshold {threshold:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_roc(y_true, y_proba, title: str, out_path: Path) -> Path | None:
    """ROC curve, or None when the split holds a single class (then the curve
    does not exist and no figure is drawn rather than an empty one)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(fpr, tpr, label=f"DNN (AUC = {roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title=title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_pred_vs_actual(y_true, y_pred, title: str, out_path: Path) -> Path:
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true, y_pred, s=14, alpha=0.6)
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", label="perfect prediction")
    ax.set(xlabel="actual lap time (s)", ylabel="predicted lap time (s)", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
