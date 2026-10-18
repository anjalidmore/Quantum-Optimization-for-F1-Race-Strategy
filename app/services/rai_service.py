"""
app.services.rai_service
========================

Responsible-AI view of the three models the dashboard compares — Simple (the
linear model in Task 6's line-up), ML (Task 6's selected model) and DNN
(Task 7) — on the identical chronological holdout every pipeline uses.

Inference only. Every model is loaded from ``artifacts/models/``; nothing is
trained or re-fit here. Predictions are recomputed on the held-out laps because
the pipelines persist per-model *metrics*, not per-lap predictions, and group
metrics and confidence distributions need the latter. ``check`` in the payload
records each model's recomputed headline metric so it can be compared with the
committed value.

What this deliberately does not do:

* **No demographic fairness.** The data has no protected attributes (age,
  gender, nationality). The groups used are the ones Task 8's stratification
  already justified — Team, Compound, Driver — and are reported as
  *performance stratification*, not protected-class fairness.
* **No invented confidence.** A model's score is reported only where the model
  natively produces one (``predict_proba``, a sigmoid output, or tree spread).
  The DNN's lap-time uncertainty uses Monte Carlo dropout, which is justified
  because the network was trained with dropout layers. Everything else is
  ``available: False`` with the reason.
"""
from __future__ import annotations

import json
import logging
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from app.core.paths import DL_MODELS_DIR, ML_METRICS_DIR
from app.intelligence.xai import stratification
from app.intelligence.xai.loading import ExplainerUnavailableError, load_target
from app.services.model_cache import ModelUnavailableError, get_model_cache

log = logging.getLogger("services.rai")

SIMPLE_MODEL = {"target_laptime": "linear_regression", "target_pit_next_lap": "logistic_regression"}
MODEL_KEYS = ("simple", "ml", "dnn")

# Monte Carlo dropout: passes per lap. 100 keeps the std estimate stable to
# roughly ±7% while the whole holdout still runs in about a second.
MC_PASSES = 100
MC_SEED = 42

# A classification call is flagged for review when the score sits this close
# to the model's own decision threshold. Each model is measured against its
# own threshold, never a shared 0.5, because the thresholds differ.
NEAR_THRESHOLD = 0.10

# A group is highlighted when its error is this many times the model's overall
# error (regression) or its false-positive rate is this far above overall.
DISPARITY_RATIO = 1.5
DISPARITY_ABS_RATE = 0.10

HIST_BINS = 10


class RaiUnavailableError(RuntimeError):
    pass


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe rows (NaN -> None)."""
    return [{k: (_f(v) if isinstance(v, (float, np.floating)) else
                 int(v) if isinstance(v, (np.integer,)) else v)
             for k, v in row.items()} for row in df.to_dict(orient="records")]


def _classical_threshold(model_name: str) -> float | None:
    path = ML_METRICS_DIR / "classification_metrics.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("models", {}).get(model_name, {})
    return _f((entry.get("test_metrics") or {}).get("decision_threshold"))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def _classical_scores(pipeline, X: pd.DataFrame, task: str) -> tuple[np.ndarray, str | None]:
    """(prediction or P(pit), score mechanism). For classification the
    returned array is a probability only if the estimator natively has one."""
    if task == "regression":
        return np.asarray(pipeline.predict(X), dtype=float).ravel(), None
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float), "predict_proba"
    return np.asarray(pipeline.predict(X), dtype=float).ravel(), None


def _tree_spread(pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Per-lap std across a random forest's trees — its native disagreement
    signal. Only for forests; any other regressor has no such signal."""
    est = pipeline.named_steps.get("model")
    if est is None or not hasattr(est, "estimators_") or type(est).__name__ != "RandomForestRegressor":
        return None
    Xt = pipeline.named_steps["preprocess"].transform(X)
    per_tree = np.stack([t.predict(Xt) for t in est.estimators_])
    return per_tree.std(axis=0)


def _mc_dropout(target: str, X: np.ndarray, numeric_mask: np.ndarray) -> np.ndarray | None:
    """MC dropout samples, shape (MC_PASSES, n_laps), in the target's real units."""
    import keras

    from app.intelligence.dl import persistence as dl_persistence

    try:
        model, scaler, y_scaler, _ = dl_persistence.load(target, DL_MODELS_DIR)
    except FileNotFoundError:
        return None
    if not any(type(layer).__name__ == "Dropout" for layer in model.layers):
        return None  # MC dropout has no meaning without dropout layers

    Xs = X.astype("float32").copy()
    if numeric_mask.any():
        Xs[:, numeric_mask] = scaler.transform(X[:, numeric_mask])
    keras.utils.set_random_seed(MC_SEED)
    samples = []
    for _ in range(MC_PASSES):
        out = keras.ops.convert_to_numpy(model(Xs, training=True)).ravel()
        if y_scaler is not None:
            out = y_scaler.inverse_transform(out.reshape(-1, 1)).ravel()
        samples.append(out)
    return np.stack(samples)


# ---------------------------------------------------------------------------
# Fairness / stratification
# ---------------------------------------------------------------------------
def _fairness(task: str, ids: pd.DataFrame, y: np.ndarray, preds: dict, thresholds: dict) -> dict:
    rows = []
    for key, p in preds.items():
        if p is None:
            continue
        df = stratification.stratify(task, ids, y, p, thresholds.get(key))
        df.insert(0, "model", key)
        # Actual pit rate per group — the base rate parity metrics are read against.
        if task == "classification":
            frame = ids.reset_index(drop=True).assign(_y=np.asarray(y, dtype=float))
            actual = {("overall", "all test laps"): float(frame["_y"].mean())}
            for g in stratification.GROUPS:
                if g in frame.columns:
                    for v, sub in frame.groupby(g):
                        actual[(g, v)] = float(sub["_y"].mean())
            df["actual_pit_rate"] = [actual.get((gt, gv)) for gt, gv in zip(df.group_type, df.group)]
        rows.append(df)
    with warnings.catch_warnings():
        # Classification rows carry all-None columns (e.g. recall for a group with
        # no pit laps); pandas warns about future dtype inference for those, which
        # does not affect values that are reported as None either way.
        warnings.simplefilter("ignore", FutureWarning)
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    summary: dict[str, dict] = {}
    flags: list[dict] = []
    for key in preds:
        if preds[key] is None:
            continue
        sub = table[table.model == key]
        overall = sub[sub.group_type == "overall"].iloc[0]
        summary[key] = {}
        for g in stratification.GROUPS:
            grp = sub[sub.group_type == g]
            if grp.empty:
                continue
            adequate = grp[grp.n_laps >= stratification.MIN_ROWS]
            entry: dict = {"n_groups": int(len(grp)), "n_adequate_groups": int(len(adequate)),
                           "min_rows": stratification.MIN_ROWS}
            if task == "regression":
                entry["mae_gap"] = _gap(adequate, "mae")
                entry["bias_gap"] = _gap(adequate, "mean_error_bias")
                entry["mae_gap_all_groups"] = _gap(grp, "mae")
            else:
                entry["demographic_parity_difference"] = _gap(adequate, "predicted_pit_rate")
                entry["false_positive_rate_gap"] = _gap(adequate, "false_positive_rate")
                with_pos = grp[grp.n_pit_laps >= stratification.MIN_POSITIVES]
                entry["equal_opportunity_difference"] = _gap(with_pos, "recall")
                entry["groups_with_enough_pit_laps"] = int(len(with_pos))
                entry["min_positives"] = stratification.MIN_POSITIVES
            summary[key][g] = entry

            # Disparity flags, marked by whether the sample can support them.
            for _, r in grp.iterrows():
                adequate_n = r.n_laps >= stratification.MIN_ROWS
                if task == "regression" and _f(r.mae) and _f(overall.mae):
                    ratio = r.mae / overall.mae
                    if ratio >= DISPARITY_RATIO:
                        flags.append({"model": key, "group_type": g, "group": r.group, "metric": "mae",
                                      "value": _f(r.mae), "overall": _f(overall.mae), "ratio": round(float(ratio), 2),
                                      "n_laps": int(r.n_laps), "adequate_sample": bool(adequate_n)})
                elif task == "classification":
                    fpr, ofpr = _f(r.false_positive_rate), _f(overall.false_positive_rate)
                    if fpr is not None and ofpr is not None and fpr - ofpr >= DISPARITY_ABS_RATE:
                        flags.append({"model": key, "group_type": g, "group": r.group,
                                      "metric": "false_positive_rate", "value": fpr, "overall": ofpr,
                                      "difference": round(float(fpr - ofpr), 4), "n_laps": int(r.n_laps),
                                      "adequate_sample": bool(adequate_n)})
    flags.sort(key=lambda f: (not f["adequate_sample"], -(f.get("ratio") or f.get("difference") or 0)))
    return {
        "groups": [g for g in stratification.GROUPS if g in ids.columns],
        "rows": _records(table),
        "summary": summary,
        "flags": flags,
        "rules": {
            "min_rows_per_group": stratification.MIN_ROWS,
            "min_pit_laps_per_group": stratification.MIN_POSITIVES,
            "disparity_ratio": DISPARITY_RATIO,
            "disparity_abs_rate": DISPARITY_ABS_RATE,
        },
        "protected_attributes_available": False,
        "note": ("The dataset holds no demographic attributes, so protected-class fairness cannot be "
                 "measured. Groups are Team, tyre Compound and Driver: operational groups a race "
                 "engineer would care about, not protected classes. Every lap is from one circuit."),
    }


def _gap(df: pd.DataFrame, col: str) -> dict | None:
    vals = df[[col, "group"]].dropna()
    if len(vals) < 2:
        return None
    hi, lo = vals.loc[vals[col].idxmax()], vals.loc[vals[col].idxmin()]
    return {"value": float(hi[col] - lo[col]), "max_group": hi.group, "max": float(hi[col]),
            "min_group": lo.group, "min": float(lo[col]), "n_groups": int(len(vals))}


# ---------------------------------------------------------------------------
# Confidence / uncertainty
# ---------------------------------------------------------------------------
def _hist(values: np.ndarray, lo: float, hi: float, bins: int = HIST_BINS, mask: np.ndarray | None = None) -> list[dict]:
    edges = np.linspace(lo, hi, bins + 1)
    out = []
    for i in range(bins):
        last = i == bins - 1
        sel = (values >= edges[i]) & ((values <= edges[i + 1]) if last else (values < edges[i + 1]))
        row = {"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(sel.sum())}
        if mask is not None:
            row["correct"] = int((sel & mask).sum())
            row["incorrect"] = int((sel & ~mask).sum())
        out.append(row)
    return out


def _classification_confidence(y: np.ndarray, preds: dict, thresholds: dict, mechanisms: dict, ids: pd.DataFrame) -> dict:
    y = y.astype(int)
    n_pos = int(y.sum())
    models = {}
    calls = {}
    for key, p in preds.items():
        t = thresholds.get(key)
        if p is None or mechanisms.get(key) is None or t is None:
            models[key] = {"available": False,
                           "reason": "This model has no native probability output or decision threshold."}
            continue
        yhat = (p >= t).astype(int)
        calls[key] = yhat
        correct = yhat == y
        near = np.abs(p - t) <= NEAR_THRESHOLD
        brier = float(np.mean((p - y) ** 2))
        calibration = None
        calibration_reason = None
        if n_pos >= stratification.MIN_POSITIVES:
            bins = []
            edges = np.linspace(0, 1, HIST_BINS + 1)
            for i in range(HIST_BINS):
                sel = (p >= edges[i]) & (p <= edges[i + 1] if i == HIST_BINS - 1 else p < edges[i + 1])
                if sel.sum():
                    bins.append({"mean_predicted": float(p[sel].mean()), "observed_rate": float(y[sel].mean()),
                                 "count": int(sel.sum())})
            calibration = bins
        else:
            calibration_reason = (f"The held-out laps contain {n_pos} real pit stop(s); a calibration curve needs "
                                  f"at least {stratification.MIN_POSITIVES} to say anything about how often a "
                                  f"predicted probability comes true.")
        models[key] = {
            "available": True,
            "mechanism": mechanisms[key],
            "threshold": t,
            "histogram": _hist(p, 0.0, 1.0, mask=correct),
            "n_near_threshold": int(near.sum()),
            "accuracy_near_threshold": _f(correct[near].mean()) if near.any() else None,
            "accuracy_far_from_threshold": _f(correct[~near].mean()) if (~near).any() else None,
            "n_predicted_pit": int(yhat.sum()),
            "brier_score": brier,
            "calibration": calibration,
            "calibration_reason": calibration_reason,
        }

    # Laps worth a human look: near any model's threshold, or the models disagree.
    review = []
    keys = [k for k in calls]
    for i in range(len(y)):
        reasons = []
        for k in keys:
            if abs(preds[k][i] - thresholds[k]) <= NEAR_THRESHOLD:
                reasons.append(f"{k}_near_threshold")
        if len(keys) > 1 and len({int(calls[k][i]) for k in keys}) > 1:
            reasons.append("models_disagree")
        if reasons:
            review.append(_lap_row(ids, i, y[i], {k: _f(preds[k][i]) for k in keys},
                                   {k: int(calls[k][i]) for k in keys}, reasons))
    review.sort(key=lambda r: (-len(r["reasons"]), -r["actual"]))
    return {
        "kind": "probability",
        "n_test": int(len(y)),
        "n_positive": n_pos,
        "near_threshold": NEAR_THRESHOLD,
        "models": models,
        "review": review,
        "n_review": len(review),
    }


def _regression_confidence(target: str, y: np.ndarray, preds: dict, spreads: dict, ids: pd.DataFrame, mc: np.ndarray | None) -> dict:
    models = {}
    err = {k: np.abs(p - y) for k, p in preds.items() if p is not None}
    emax = float(max(e.max() for e in err.values())) if err else 1.0
    for key, p in preds.items():
        if p is None:
            models[key] = {"available": False, "reason": "No predictions: model not found."}
            continue
        e = err[key]
        base = {"mae": float(e.mean()), "error_histogram": _hist(e, 0.0, emax)}
        std = spreads.get(key)
        if std is None:
            models[key] = {**base, "available": False, "reason": spreads.get(f"{key}_reason")}
            continue
        z = e / np.maximum(std, 1e-9)
        cut = float(np.quantile(std, 0.9))
        order = np.argsort(std)
        tercile = np.array_split(order, 3)
        models[key] = {
            **base,
            "available": True,
            "mechanism": spreads.get(f"{key}_mechanism"),
            "std_histogram": _hist(std, 0.0, float(std.max())),
            "mean_std": float(std.mean()),
            # If the spread were a calibrated Gaussian sigma these would be ~68% and ~95%.
            "coverage_1sigma": float((z <= 1).mean()),
            "coverage_2sigma": float((z <= 2).mean()),
            "spearman_std_vs_error": float(pd.Series(std).corr(pd.Series(e), method="spearman")),
            "error_by_uncertainty_tercile": [
                {"label": lbl, "mean_std": float(std[idx].mean()), "mae": float(e[idx].mean()), "n": int(len(idx))}
                for lbl, idx in zip(("lowest third", "middle third", "highest third"), tercile)
            ],
            "flag_cutoff_std": cut,
            "n_flagged": int((std >= cut).sum()),
        }
    review = []
    for key in MODEL_KEYS:
        m = models.get(key)
        if not m or not m.get("available"):
            continue
        std = spreads[key]
        for i in np.where(std >= m["flag_cutoff_std"])[0]:
            review.append({**_lap_row(ids, int(i), float(y[i]), {k: _f(p[i]) for k, p in preds.items() if p is not None},
                                      None, [f"{key}_high_uncertainty"]),
                           "std": float(std[i]), "abs_error": float(err[key][i])})
    review.sort(key=lambda r: -r["std"])
    return {"kind": "interval", "n_test": int(len(y)), "models": models, "review": review, "n_review": len(review),
            "mc_passes": MC_PASSES if mc is not None else None}


def _lap_row(ids: pd.DataFrame, i: int, actual, scores: dict, calls: dict | None, reasons: list[str]) -> dict:
    r = ids.iloc[i]
    return {"row": i, "driver": r.get("Driver"), "team": r.get("Team"), "compound": r.get("Compound"),
            "lap": int(r["LapNumber"]) if "LapNumber" in r and pd.notna(r["LapNumber"]) else None,
            "actual": _f(actual), "scores": scores, "calls": calls, "reasons": reasons}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def summary(target: str) -> dict:
    try:
        t = load_target(target)
    except ExplainerUnavailableError as exc:
        raise RaiUnavailableError(str(exc)) from exc

    cache = get_model_cache()
    X_df = pd.DataFrame(t.X_test, columns=t.features)
    ids = t.test_ids.reset_index(drop=True)
    y = np.asarray(t.y_test, dtype=float)

    names: dict[str, str | None] = {"simple": SIMPLE_MODEL[target], "ml": t.classical_name, "dnn": "dnn_mlp"}
    preds: dict[str, np.ndarray | None] = {}
    mechanisms: dict[str, str | None] = {}
    pipelines = {}
    for key in ("simple", "ml"):
        try:
            _, pipe = cache.get_pipeline(target, names[key])
        except ModelUnavailableError as exc:
            log.warning("RAI: %s model unavailable: %s", key, exc)
            preds[key], mechanisms[key] = None, None
            continue
        pipelines[key] = pipe
        preds[key], mechanisms[key] = _classical_scores(pipe, X_df, t.task)
    preds["dnn"] = np.asarray(t.dnn_predict(t.X_test), dtype=float).ravel()
    mechanisms["dnn"] = "sigmoid output" if t.task == "classification" else None

    thresholds = {}
    if t.task == "classification":
        thresholds = {"simple": _classical_threshold(names["simple"]), "ml": _classical_threshold(names["ml"]),
                      "dnn": _f(t.decision_threshold)}

    # Recomputed headline metric, for comparison with the committed artifacts.
    check = {}
    for key, p in preds.items():
        if p is None:
            continue
        if t.task == "regression":
            check[key] = {"mae": float(np.mean(np.abs(p - y)))}
        else:
            from sklearn.metrics import average_precision_score
            check[key] = {"pr_auc": float(average_precision_score(y, p)) if y.sum() else None}

    if t.task == "classification":
        confidence = _classification_confidence(y, preds, thresholds, mechanisms, ids)
    else:
        spreads: dict = {}
        for key in ("simple", "ml"):
            pipe = pipelines.get(key)
            s = _tree_spread(pipe, X_df) if pipe is not None else None
            if s is not None:
                spreads[key], spreads[f"{key}_mechanism"] = s, "random-forest tree spread (std across trees)"
            else:
                spreads[f"{key}_reason"] = _no_interval_reason(names[key], pipe, t)
        mc = _mc_dropout(target, t.X_test, t.numeric_mask)
        if mc is not None:
            spreads["dnn"] = mc.std(axis=0)
            spreads["dnn_mechanism"] = f"Monte Carlo dropout ({MC_PASSES} stochastic passes)"
        else:
            spreads["dnn_reason"] = "The saved network has no dropout layers, so MC dropout does not apply."
        confidence = _regression_confidence(target, y, preds, spreads, ids, mc)

    return {
        "target": target,
        "task": t.task,
        "n_test": int(len(y)),
        "n_positive": int(y.sum()) if t.task == "classification" else None,
        "models": {k: {"name": names[k], "available": preds.get(k) is not None,
                       "threshold": thresholds.get(k)} for k in MODEL_KEYS},
        "check": check,
        "fairness": _fairness(t.task, ids, y, preds, thresholds),
        "confidence": confidence,
        "source_dataset": t.source_dataset,
    }


def _no_interval_reason(model: str | None, pipeline, t) -> str:
    if model == "linear_regression" and pipeline is not None:
        Xt = pipeline.named_steps["preprocess"].transform(pd.DataFrame(t.X_train, columns=t.features))
        Xt = np.column_stack([np.ones(len(Xt)), np.asarray(Xt, dtype=float)])
        rank = int(np.linalg.matrix_rank(Xt))
        if rank < Xt.shape[1]:
            return (f"Ordinary least squares prediction intervals need an invertible XᵀX. Here the design matrix "
                    f"has rank {rank} for {Xt.shape[1]} columns: the one-hot driver and team columns are "
                    f"collinear, so that inverse does not exist. Not available.")
        return ("Ordinary least squares supports prediction intervals, but the saved pipeline records no residual "
                "variance to build them from. Not available.")
    if model == "svr":
        return "Support vector regression outputs a point estimate only; it has no native uncertainty. Not available."
    return f"{model} produces a point estimate with no native uncertainty. Not available."
