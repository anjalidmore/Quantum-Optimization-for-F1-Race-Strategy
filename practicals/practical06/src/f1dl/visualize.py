"""
f1dl.visualize
==============

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
