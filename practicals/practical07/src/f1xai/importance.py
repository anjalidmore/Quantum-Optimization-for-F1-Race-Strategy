"""
f1xai.importance
================

Global feature importance by **permutation**.

Why permutation importance rather than a model's built-in ``feature_importances_``?
Because Task 8 must compare a random forest against a Keras network, and only a
model-agnostic method gives numbers that mean the same thing for both. A tree's
Gini importance and a network's gradient magnitudes are not on a comparable
scale and are not even measuring the same quantity.

Permutation importance asks one question of any model: *if I destroy the
relationship between this feature and the target by shuffling it, how much
worse does the model get?* The answer is in units of the metric, so it is
directly interpretable and directly comparable across model families.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score

RANDOM_STATE = 42


def _score(task: str, y_true: np.ndarray, pred: np.ndarray) -> float | None:
    if task == "regression":
        return float(mean_absolute_error(y_true, pred))
    if len(np.unique(y_true)) < 2:
        return None  # ROC-AUC undefined; caller reports this rather than faking it
    return float(roc_auc_score(y_true, pred))


def permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task: str,
    n_repeats: int = 10,
    seed: int = RANDOM_STATE,
) -> list[dict]:
    """Mean drop in score when each column is shuffled, over ``n_repeats``.

    For regression the score is MAE (lower is better), so importance is
    ``permuted_MAE - baseline_MAE`` — a positive number means shuffling made
    the model worse, i.e. the feature was being used.

    For classification the score is ROC-AUC (higher is better), so importance
    is ``baseline_AUC - permuted_AUC``, giving the same sign convention.
    """
    rng = np.random.default_rng(seed)
    baseline = _score(task, y, predict_fn(X))
    if baseline is None:
        return [
            {"feature": f, "importance": None, "std": None,
             "undefined_reason": "ROC-AUC undefined on this split (single class present)."}
            for f in feature_names
        ]

    results = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            s = _score(task, y, predict_fn(Xp))
            if s is None:
                continue
            drops.append(s - baseline if task == "regression" else baseline - s)
        results.append({
            "feature": name,
            "importance": float(np.mean(drops)) if drops else None,
            "std": float(np.std(drops)) if drops else None,
            "undefined_reason": None,
        })

    results.sort(key=lambda r: (r["importance"] is not None, r["importance"] or 0.0), reverse=True)
    return results


def compare_importance(dnn_rows: list[dict], classical_rows: list[dict]) -> list[dict]:
    """Align two importance rankings feature-by-feature so the report can show
    where the two model families agree and where they disagree."""
    d = {r["feature"]: r for r in dnn_rows}
    c = {r["feature"]: r for r in classical_rows}
    d_rank = {r["feature"]: i + 1 for i, r in enumerate(dnn_rows)}
    c_rank = {r["feature"]: i + 1 for i, r in enumerate(classical_rows)}

    out = []
    for f in d:
        out.append({
            "feature": f,
            "dnn_importance": d[f]["importance"],
            "classical_importance": c.get(f, {}).get("importance"),
            "dnn_rank": d_rank[f],
            "classical_rank": c_rank.get(f),
            "rank_gap": (abs(d_rank[f] - c_rank[f]) if f in c_rank else None),
        })
    out.sort(key=lambda r: r["dnn_rank"])
    return out
