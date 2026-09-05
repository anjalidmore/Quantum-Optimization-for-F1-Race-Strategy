"""
app.intelligence.xai.lime_analysis
=================================

LIME local explanations, computed on the same rows SHAP explains so the two
can be compared directly.

**How LIME differs from SHAP** - this distinction matters and is easy to blur:

* **SHAP** is *game-theoretic attribution*. It distributes the gap between
  this prediction and the average prediction among the features, using
  Shapley values, and those values provably sum to that gap (local accuracy).
* **LIME** fits a *local surrogate*. It perturbs the row, asks the real model
  what it predicts for each perturbation, and fits a simple weighted linear
  model to those answers in the neighbourhood of the row. The coefficients of
  that surrogate are the explanation.

So SHAP answers "how is the credit for this prediction fairly divided?" while
LIME answers "what simple model behaves like the real one *around here*?"
They usually agree on which features matter; when they disagree, that is
information - it means the local decision surface is not well approximated by
a line, and the trust score treats it as such.

LIME is also stochastic: it samples perturbations. ``random_state`` is fixed
so this practical's output is reproducible.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

RANDOM_STATE = 42
NUM_SAMPLES = 2000


def build_explainer(
    X_train: np.ndarray, feature_names: list[str], task: str, seed: int = RANDOM_STATE
) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=list(feature_names),
        mode="regression" if task == "regression" else "classification",
        class_names=["stay_out", "pit_now"] if task == "classification" else None,
        discretize_continuous=True,
        random_state=seed,
    )


def explain_row(
    explainer: LimeTabularExplainer,
    row: np.ndarray,
    predict_fn,
    task: str,
    num_features: int = 5,
    num_samples: int = NUM_SAMPLES,
) -> dict:
    """Explain one row. ``predict_fn`` takes a 2-D array and returns a 1-D
    array of predictions (regression) or positive-class probabilities
    (classification); LIME's classification mode needs both columns, so the
    probability is expanded here rather than in every caller."""
    if task == "classification":
        def wrapped(rows):
            p = np.asarray(predict_fn(np.asarray(rows, dtype="float32"))).ravel()
            return np.column_stack([1.0 - p, p])
    else:
        def wrapped(rows):
            return np.asarray(predict_fn(np.asarray(rows, dtype="float32"))).ravel()

    kwargs = dict(num_features=num_features, num_samples=num_samples)
    if task == "classification":
        kwargs["labels"] = (1,)

    exp = explainer.explain_instance(np.asarray(row, dtype=float), wrapped, **kwargs)
    # LIME stores regression explanations under label 1 by convention and
    # refuses ``available_labels()`` outside classification mode, so the label
    # is fixed here rather than queried.
    label = 1
    pairs = exp.as_list(label=label)

    return {
        "explainer": "LimeTabularExplainer",
        "num_samples": num_samples,
        "contributions": [
            {"condition": cond, "weight": float(w),
             "direction": "increases" if w > 0 else "decreases"}
            for cond, w in pairs
        ],
        "intercept": float(exp.intercept[label]),
        "local_r2": float(exp.score),
        "explanation": exp,
        "note": (
            "Weights are coefficients of a local linear surrogate fitted to "
            f"{num_samples} perturbations around this row, not Shapley values. "
            "`local_r2` reports how well that surrogate reproduced the real model "
            "nearby - a low value means the linear explanation is a poor stand-in."
        ),
    }


def top_features(result: dict, feature_names: list[str], top_k: int = 3) -> list[str]:
    """Map LIME's discretised conditions (``tyre_life > 12.00``) back to plain
    feature names, so SHAP's and LIME's top-k lists can be compared."""
    out: list[str] = []
    for c in result["contributions"]:
        cond = c["condition"]
        match = next((f for f in feature_names if f in cond), None)
        if match and match not in out:
            out.append(match)
        if len(out) == top_k:
            break
    return out
