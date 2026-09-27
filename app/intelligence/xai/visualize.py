"""
app.intelligence.xai.visualize
==============================

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


def shap_bar(ranking: list[dict], title: str, out_path: Path, top_n: int = 15) -> Path:
    """Global SHAP feature importance: mean |SHAP value| per feature."""
    rows = ranking[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.6))
    ax.barh([r["feature"] for r in rows], [r["mean_abs_shap"] for r in rows], color="#c0392b")
    ax.set(xlabel="mean |SHAP value| (average impact on the model output)", title=title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


def feature_importance_overview(per_target: dict, out_path: Path, top_n: int = 12) -> Path:
    """One figure, one panel per target: permutation importance of the Task 7
    DNN on the chronological test laps (the performance lost when a feature is
    shuffled)."""
    fig, axes = plt.subplots(1, len(per_target), figsize=(7 * len(per_target), 5.6))
    axes = np.atleast_1d(axes)
    for ax, (target, rows) in zip(axes, per_target.items()):
        rows = [r for r in rows if r.get("importance") is not None][:top_n][::-1]
        ax.barh([r["feature"] for r in rows], [r["importance"] for r in rows],
                xerr=[r.get("std") or 0 for r in rows], color="#3b6ea5", capsize=2)
        ax.axvline(0, color="grey", lw=0.8)
        metric = "MAE increase (s)" if target == "target_laptime" else "ROC-AUC drop"
        ax.set(title=f"{target} — Task 7 DNN", xlabel=f"permutation importance: {metric}")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Task 8 — global feature importance (test laps)")
    fig.tight_layout()
    return _save(fig, out_path)


def stratification_plot(strata, task: str, title: str, out_path: Path) -> Path:
    """Per-driver / team / compound error (regression) or recall (classification),
    with small groups hatched so they are not read as findings."""
    import pandas as pd

    df = pd.DataFrame(strata)
    df = df[df.group_type != "overall"]
    metric = "mae" if task == "regression" else "recall"
    types = [g for g in ("Team", "Compound", "Driver") if g in set(df.group_type)]
    fig, axes = plt.subplots(1, len(types), figsize=(5.2 * len(types), 5.2), squeeze=False)
    for ax, g in zip(axes[0], types):
        sub = df[df.group_type == g].sort_values("group")
        vals = sub[metric].astype(float).fillna(0).to_numpy()
        small = sub.sample_note.str.startswith(("SMALL", "INSUFFICIENT")).to_numpy()
        bars = ax.barh(sub.group.astype(str), vals, color=["#bbbbbb" if s else "#3b6ea5" for s in small])
        for b, s in zip(bars, small):
            if s:
                b.set_hatch("//")
        ax.set(title=g, xlabel=("MAE (s)" if metric == "mae" else "recall on pit laps"))
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title + "\n(grey hatched = too few laps to compare)")
    fig.tight_layout()
    return _save(fig, out_path)
