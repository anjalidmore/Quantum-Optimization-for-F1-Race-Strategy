"""
app.intelligence.qml.baselines
==============================

Classical models to compare the quantum ones against, at two different levels
of fairness — because "quantum vs classical" means nothing without saying
*which* classical.

**1. The project's best classical model.** Task 6's selected model, read from
its committed metrics. It uses all 45 (or 8) selected features and hundreds or
thousands of effective parameters, so it answers "is the quantum model
competitive with what we already have?" — the honest headline comparison, and
not a like-for-like one.

**2. A parameter-matched model on the same reduced inputs.** Trained here, on
the identical PCA-reduced angle-encoded matrix the circuits see, with a
comparable number of trainable parameters. This is the like-for-like
comparison: same information, same budget, different model family. Two of them
are fitted:

* logistic / linear regression — the smallest sensible model (n_features + 1);
* a tiny MLP with one hidden layer sized so its parameter count lands near the
  circuit's.

Every parameter count below is counted from the fitted object, never asserted.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from app.core.paths import ML_METRICS_DIR


def task6_reference(task: str) -> dict:
    """Task 6's selected model and its committed test metrics, read from disk."""
    path = ML_METRICS_DIR / ("classification_metrics.json" if task == "classification" else "regression_metrics.json")
    if not path.exists():
        return {"available": False, "reason": f"{path.name} not found; run Task 6 first"}
    blob = json.loads(path.read_text())
    name = blob["best_model"]
    entry = blob["models"][name]
    return {
        "available": True,
        "model": name,
        "source": f"artifacts/metrics/{path.name}",
        "features_used": len(blob.get("features", [])) or None,
        "test_metrics": entry["test_metrics"],
        "cv_summary": entry.get("cv_summary", {}),
        "decision_threshold": (entry.get("threshold") or {}).get("threshold"),
    }


def _mlp_hidden_for(target_parameters: int, n_features: int, task: str) -> int:
    """Hidden width whose parameter count is closest to the circuit's.

    An MLP with one hidden layer of h units has
    ``h*(n_features + 1) + (h + 1)`` parameters for a single output.
    """
    best_h, best_gap = 1, None
    for h in range(1, 33):
        params = h * (n_features + 1) + (h + 1)
        gap = abs(params - target_parameters)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


def count_mlp_parameters(mlp) -> int:
    return int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))


def train_matched_classifier(X, y, *, target_parameters: int, seed: int = 42) -> dict:
    """Logistic regression and a parameter-matched tiny MLP, on the circuit's inputs."""
    X = np.asarray(X, dtype=float)
    out: dict[str, dict] = {}

    t0 = time.perf_counter()
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(X, y)
    out["logistic_regression"] = {
        "model": logreg,
        "train_seconds": time.perf_counter() - t0,
        "n_parameters": int(logreg.coef_.size + logreg.intercept_.size),
        "note": "smallest sensible classical model on the same reduced inputs",
    }

    h = _mlp_hidden_for(target_parameters, X.shape[1], "classification")
    t0 = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(h,), max_iter=4000, random_state=seed, learning_rate_init=0.05,
    ).fit(X, y)
    out["tiny_mlp"] = {
        "model": mlp,
        "train_seconds": time.perf_counter() - t0,
        "n_parameters": count_mlp_parameters(mlp),
        "hidden_units": h,
        "note": f"hidden layer sized to land near the circuit's {target_parameters} parameters",
    }
    return out


def train_matched_regressor(X, y, *, target_parameters: int, seed: int = 42) -> dict:
    """Linear regression and a parameter-matched tiny MLP, on the circuit's inputs."""
    X = np.asarray(X, dtype=float)
    out: dict[str, dict] = {}

    t0 = time.perf_counter()
    linear = LinearRegression().fit(X, y)
    out["linear_regression"] = {
        "model": linear,
        "train_seconds": time.perf_counter() - t0,
        "n_parameters": int(np.size(linear.coef_) + 1),
        "note": "smallest sensible classical model on the same reduced inputs",
    }

    h = _mlp_hidden_for(target_parameters, X.shape[1], "regression")
    t0 = time.perf_counter()
    mlp = MLPRegressor(
        hidden_layer_sizes=(h,), max_iter=4000, random_state=seed, learning_rate_init=0.05,
    ).fit(X, y)
    out["tiny_mlp"] = {
        "model": mlp,
        "train_seconds": time.perf_counter() - t0,
        "n_parameters": count_mlp_parameters(mlp),
        "hidden_units": h,
        "note": f"hidden layer sized to land near the circuit's {target_parameters} parameters",
    }
    return out


def proba(fitted: dict, X) -> np.ndarray:
    return np.asarray(fitted["model"].predict_proba(np.asarray(X, dtype=float))[:, 1], dtype=float)


def predict(fitted: dict, X) -> np.ndarray:
    return np.asarray(fitted["model"].predict(np.asarray(X, dtype=float)), dtype=float)
