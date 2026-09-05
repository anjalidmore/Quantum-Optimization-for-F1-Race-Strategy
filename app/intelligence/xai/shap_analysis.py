"""
app.intelligence.xai.shap_analysis
=================================

SHAP attributions for both model families.

**Which explainer, and why.**

* ``TreeExplainer`` for the random forest. It is *exact* for tree ensembles -
  it walks the trees and computes Shapley values in polynomial time rather
  than approximating them. Free accuracy; always prefer it when the model is
  a tree.
* ``KernelExplainer`` for the Keras network. ``DeepExplainer`` exists, but its
  Keras 3 support targets the TensorFlow backend and this project runs Keras
  on PyTorch (TensorFlow has no Python 3.14 wheel). ``KernelExplainer`` is
  fully model-agnostic - it only needs a ``predict`` function - so it works
  here unchanged.

**The trade-off, stated plainly.** ``KernelExplainer`` *approximates* Shapley
values by sampling coalitions, and its cost grows with the number of
background samples and ``nsamples``. The background set is therefore
summarised with ``shap.kmeans`` rather than passing the whole training set,
and ``nsamples`` is fixed and recorded in the report. Two consequences an
evaluator should know: the DNN's SHAP values carry sampling noise the forest's
do not, and they are not directly comparable in magnitude to the tree's exact
values - only in rank and sign.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import shap  # noqa: E402

# KernelExplainer logs its internal coalition weights at INFO level, which
# drowns this practical's own progress output. It is diagnostic noise, not a
# result, so it is silenced here rather than by lowering the global log level.
import logging  # noqa: E402

logging.getLogger("shap").setLevel(logging.WARNING)

RANDOM_STATE = 42
KERNEL_NSAMPLES = 200
BACKGROUND_K = 25


def tree_shap(estimator, X: np.ndarray, feature_names: list[str], task: str) -> dict:
    """Exact Shapley values for a tree ensemble."""
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X, check_additivity=False)
    values = _to_positive_class(values, task)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = np.asarray(base).ravel()
        base = float(base[-1] if len(base) > 1 else base[0])
    return {
        "explainer": "TreeExplainer",
        "exact": True,
        "values": np.asarray(values, dtype=float),
        "base_value": float(base),
        "feature_names": feature_names,
        "note": "Exact Shapley values; TreeExplainer enumerates the ensemble rather than sampling.",
    }


def kernel_shap(
    predict_fn,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    nsamples: int = KERNEL_NSAMPLES,
    k: int = BACKGROUND_K,
    seed: int = RANDOM_STATE,
) -> dict:
    """Model-agnostic, sampled Shapley values for the Keras network."""
    np.random.seed(seed)
    background = shap.kmeans(X_background, min(k, len(X_background)))
    explainer = shap.KernelExplainer(lambda r: predict_fn(np.asarray(r, dtype="float32")), background)
    values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[..., -1]
    return {
        "explainer": "KernelExplainer",
        "exact": False,
        "values": values,
        "base_value": float(np.asarray(explainer.expected_value).ravel()[-1]),
        "feature_names": feature_names,
        "nsamples": nsamples,
        "background_k": min(k, len(X_background)),
        "note": (
            f"Approximate Shapley values sampled with nsamples={nsamples} over a "
            f"k-means background of {min(k, len(X_background))} points. Ranks and signs "
            f"are meaningful; magnitudes carry sampling noise and are not directly "
            f"comparable to TreeExplainer's exact values."
        ),
    }


def _to_positive_class(values, task: str):
    """Normalise SHAP's per-class output to the positive class for binary
    classification, leaving regression output untouched."""
    arr = np.asarray(values)
    if task == "classification" and arr.ndim == 3:
        return arr[..., -1]
    if task == "classification" and isinstance(values, list) and len(values) == 2:
        return np.asarray(values[1])
    return arr


def global_ranking(result: dict) -> list[dict]:
    """Mean |SHAP| per feature - the standard global summary."""
    vals = np.abs(result["values"]).mean(axis=0)
    rows = [
        {"feature": f, "mean_abs_shap": float(v)}
        for f, v in zip(result["feature_names"], vals)
    ]
    rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return rows


def explain_row(result: dict, row_index: int, top_n: int = 5) -> list[dict]:
    """Per-feature contribution for one prediction, largest magnitude first."""
    vals = result["values"][row_index]
    rows = [
        {"feature": f, "shap_value": float(v), "direction": "increases" if v > 0 else "decreases"}
        for f, v in zip(result["feature_names"], vals)
    ]
    rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    return rows[:top_n]


def stability(result: dict, row_index: int, top_k: int = 3) -> list[str]:
    """The top-k features for one row, used by the trust score to compare
    SHAP's account of a prediction against LIME's."""
    return [r["feature"] for r in explain_row(result, row_index, top_n=top_k)]
