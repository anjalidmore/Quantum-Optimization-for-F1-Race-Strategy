"""
f1xai.visualize
===============

Figures for the Task 8 deliverables: SHAP summary and waterfall plots, LIME
bar charts, permutation-importance comparisons, and the counterfactual
prediction curve.

SHAP's own plotting API is used where it exists (its summary plot is the
canonical view an evaluator will recognise); the waterfall is drawn directly
so it works uniformly for both the exact tree values and the sampled kernel
values without depending on SHAP's Explanation object plumbing.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def shap_summary(result: dict, X: np.ndarray, title: str, out_path: Path) -> Path:
    import shap
    fig = plt.figure(figsize=(8, max(3.0, 0.42 * len(result["feature_names"]) + 2)))
    shap.summary_plot(
        result["values"], X, feature_names=result["feature_names"], show=False, plot_size=None
    )
    plt.title(title, fontsize=11)
    return _save(fig, out_path)


def shap_waterfall(row: list[dict], base_value: float, prediction: float,
                   title: str, out_path: Path) -> Path:
    """Per-feature contributions for one prediction, ordered by magnitude."""
    names = [r["feature"] for r in row][::-1]
    vals = [r["shap_value"] for r in row][::-1]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 2.2))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to this prediction)")
    ax.set_title(f"{title}\nbase {base_value:.4f}  ->  prediction {prediction:.4f}", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.4f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_path)


def lime_plot(result: dict, title: str, out_path: Path) -> Path:
    conds = [c["condition"] for c in result["contributions"]][::-1]
    weights = [c["weight"] for c in result["contributions"]][::-1]
    colors = ["#c0392b" if w > 0 else "#2471a3" for w in weights]

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(conds) + 2.2))
    ax.barh(conds, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("local surrogate weight")
    ax.set_title(f"{title}\nlocal surrogate R2 = {result['local_r2']:.3f}", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_path)


def importance_comparison(rows: list[dict], title: str, out_path: Path, top_n: int = 12) -> Path:
    rows = [r for r in rows if r["dnn_importance"] is not None][:top_n][::-1]
    if not rows:
        return out_path
    names = [r["feature"] for r in rows]
    dnn = [r["dnn_importance"] for r in rows]
    cls = [r["classical_importance"] or 0.0 for r in rows]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 2.4))
    ax.barh(y - 0.2, dnn, height=0.38, label="deep network", color="#c0392b")
    ax.barh(y + 0.2, cls, height=0.38, label="classical", color="#7f8c8d")
    ax.set_yticks(y, names)
    ax.set_xlabel("permutation importance (drop in score when shuffled)")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_path)


def counterfactual_curve(cf: dict, title: str, out_path: Path) -> Path:
    grid = cf["prediction_curve"]["grid"]
    pred = cf["prediction_curve"]["prediction"]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(grid, pred, color="#2471a3", linewidth=2, label="model output")
    ax.axhline(cf["threshold"], color="grey", linestyle=":", label=f"decision threshold ({cf['threshold']:g})")
    ax.axvline(cf["original_value"], color="black", linestyle="--", linewidth=1,
               label=f"current {cf['feature']} ({cf['original_value']:.4g})")
    if cf["reachable"]:
        ax.axvline(cf["crossing_value"], color="#c0392b", linewidth=1.6,
                   label=f"flips at {cf['crossing_value']:.4g}")
    ax.set_xlabel(cf["feature"])
    ax.set_ylabel("prediction")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def fairness_plot(assessment: dict, title: str, out_path: Path) -> Path:
    shares = [assessment["identity_attribution_share"], assessment["race_state_attribution_share"]]
    labels = [
        f"identity features\n({assessment['n_identity_features']} of {assessment['n_features']})",
        f"race-state features\n({assessment['n_features'] - assessment['n_identity_features']} of {assessment['n_features']})",
    ]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(["attribution"], [shares[0]], color="#c0392b", label=labels[0])
    ax.barh(["attribution"], [shares[1]], left=[shares[0]], color="#27ae60", label=labels[1])
    ax.axvline(assessment["expected_share_if_uniform"], color="black", linestyle="--",
               linewidth=1.2, label="identity share if attribution were uniform")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of total |SHAP| attribution")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_path)
