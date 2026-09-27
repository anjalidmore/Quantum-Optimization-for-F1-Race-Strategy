"""
app.intelligence.xai.stratification
===================================

**F1-specific performance stratification.**

This is *not* a protected-attribute fairness analysis. The Task 5 data holds no
demographic information about drivers (no age, gender, nationality), so a
fairness assessment in that sense cannot be done and is not claimed. What can
be checked is whether the Task 7 model performs systematically differently for
different **drivers**, **teams** or **tyre compounds** on the held-out laps — a
model that consistently missed one team's pit stops would be less useful to
that team.

Circuit and race-condition strata are not reported: every lap in the dataset
comes from one session (2023 Bahrain GP), so there is exactly one circuit.

Small groups are reported but flagged: with ~180 held-out laps split across 20
drivers (≈9 laps each), per-driver metrics — and anything built on a handful of
pit laps — are descriptive only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ("Driver", "Team", "Compound")
MIN_ROWS = 30
MIN_POSITIVES = 5


def stratify(task: str, ids: pd.DataFrame, y_true, pred, threshold: float | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    rows = []
    frame = ids.reset_index(drop=True).assign(_y=y_true, _p=pred, _all="all test laps")
    for group in ("_all", *[g for g in GROUPS if g in frame.columns]):
        for value, sub in frame.groupby(group):
            rows.append({"group_type": "overall" if group == "_all" else group, "group": value,
                         **_metrics(task, sub["_y"].to_numpy(), sub["_p"].to_numpy(), threshold)})
    return pd.DataFrame(rows)


def _metrics(task: str, y: np.ndarray, p: np.ndarray, threshold: float | None) -> dict:
    n = len(y)
    if task == "regression":
        err = p - y
        return {"n_laps": n, "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mean_error_bias": float(np.mean(err)),
                "sample_note": "OK" if n >= MIN_ROWS else f"SMALL SAMPLE (n={n}) — descriptive only"}
    yhat = (p >= threshold).astype(int)
    y = y.astype(int)
    tp, fp = int(((y == 1) & (yhat == 1)).sum()), int(((y == 0) & (yhat == 1)).sum())
    fn, tn = int(((y == 1) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 0)).sum())
    pos, neg = tp + fn, fp + tn
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / pos if pos else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if pos and (tp + fp) else None)
    ok = n >= MIN_ROWS and pos >= MIN_POSITIVES
    return {"n_laps": n, "n_pit_laps": pos, "accuracy": (tp + tn) / n if n else None,
            "precision": prec, "recall": rec, "f1": f1,
            "false_positive_rate": fp / neg if neg else None,
            "false_negative_rate": fn / pos if pos else None,
            "predicted_pit_rate": float(yhat.mean()) if n else None,
            "mean_pit_probability": float(p.mean()) if n else None,
            "sample_note": "OK" if ok else
            f"INSUFFICIENT SAMPLE (n={n}, pit laps={pos}) — descriptive only, do not compare"}
