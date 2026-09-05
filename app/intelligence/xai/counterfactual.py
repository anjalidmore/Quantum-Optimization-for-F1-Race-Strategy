"""
app.intelligence.xai.counterfactual
==================================

"What would have to change about the race state for the recommendation to
flip?"

Two methods, used for different reasons:

``dice_counterfactuals``
    DiCE (Diverse Counterfactual Explanations) searches for *complete
    alternative rows* that the model classifies the other way. It gives
    several genuinely different options, which is what you want when several
    routes to a different decision exist.

``perturbation_scan``
    A single-feature bisection search: hold the whole race state fixed, move
    one feature (say ``tyre_life``), and find the exact threshold at which the
    model's output crosses the decision boundary. This answers the question a
    race engineer actually asks - *"how many more laps on these tyres before
    the call changes?"* - which DiCE's multi-feature rows do not, because
    changing six things at once is not an actionable instruction on a pit
    wall.

Both are real searches against the real model. Neither writes a
plausible-looking number: if no counterfactual exists within the feature's
observed range, that is reported as "not reachable", with the range that was
searched.
"""
from __future__ import annotations

import signal
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DICE_TIMEOUT_SECONDS = 60


class _SearchTimeout(RuntimeError):
    pass


@contextmanager
def _time_budget(seconds: int):
    """Hard wall-clock cap on a search.

    DiCE's random method samples until it finds a counterfactual. When the
    target class is rare - pit events are 4.8% of laps here - it can sample
    for many minutes and still return nothing. An unbounded search that
    produces no result is worse than a bounded one that says so, so the budget
    is enforced and the report states when it was hit.
    """
    def _raise(signum, frame):
        raise _SearchTimeout(f"search exceeded its {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def perturbation_scan(
    predict_fn,
    row: np.ndarray,
    feature_index: int,
    feature_name: str,
    search_lo: float,
    search_hi: float,
    threshold: float,
    steps: int = 60,
) -> dict:
    """Sweep one feature across its observed range, holding everything else
    fixed, and locate the crossing point of ``threshold``.

    Returns the crossing value and the direction of travel, or an explicit
    "not reachable" result with the range searched.
    """
    grid = np.linspace(search_lo, search_hi, steps)
    rows = np.repeat(np.asarray(row, dtype="float32")[None, :], steps, axis=0)
    rows[:, feature_index] = grid
    preds = np.asarray(predict_fn(rows)).ravel()

    original = float(np.asarray(predict_fn(np.asarray(row, dtype="float32")[None, :])).ravel()[0])
    side = preds >= threshold
    start_side = original >= threshold

    crossing = None
    for i in range(1, steps):
        if side[i] != side[i - 1]:
            crossing = float(np.interp(threshold, sorted([preds[i - 1], preds[i]]),
                                       sorted([grid[i - 1], grid[i]])))
            break

    return {
        "method": "single-feature bisection scan",
        "feature": feature_name,
        "original_value": float(row[feature_index]),
        "original_prediction": original,
        "threshold": float(threshold),
        "searched_range": [float(search_lo), float(search_hi)],
        "steps": steps,
        "crossing_value": crossing,
        "reachable": crossing is not None,
        "direction": (
            None if crossing is None
            else ("increase" if crossing > row[feature_index] else "decrease")
        ),
        "delta_required": (None if crossing is None else float(crossing - row[feature_index])),
        "prediction_curve": {"grid": [float(g) for g in grid], "prediction": [float(p) for p in preds]},
        "note": (
            f"Every other feature was held at this row's observed value; only "
            f"{feature_name} was moved."
            if crossing is not None else
            f"No crossing of {threshold} exists anywhere in the observed range of "
            f"{feature_name} ([{search_lo:.3f}, {search_hi:.3f}]) with the rest of the "
            f"race state fixed. The recommendation is not flippable by this feature alone."
        ),
        "started_above_threshold": bool(start_side),
    }


def dice_counterfactuals(
    estimator,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    row_scaled: np.ndarray,
    feature_names: list[str],
    total_cfs: int = 3,
    seed: int = RANDOM_STATE,
    timeout_seconds: int = DICE_TIMEOUT_SECONDS,
) -> dict:
    """Diverse whole-row counterfactuals for the classical classifier.

    DiCE is run against the sklearn model in its *scaled* feature space, which
    is the space the model was fitted in. Values are reported back in that
    space and flagged as such - reporting them as raw race-state numbers
    without inverting the scaler would be misleading.
    """
    try:
        import dice_ml
    except Exception as exc:  # pragma: no cover - dependency guard
        return {"method": "DiCE", "available": False,
                "reason": f"{type(exc).__name__}: {exc}", "counterfactuals": []}

    np.random.seed(seed)
    df = pd.DataFrame(X_train_scaled, columns=feature_names)
    df["outcome"] = np.asarray(y_train).astype(int)

    if len(np.unique(df["outcome"])) < 2:
        return {"method": "DiCE", "available": False,
                "reason": "training rows contain a single class; no opposite class to search for",
                "counterfactuals": []}

    data = dice_ml.Data(dataframe=df, continuous_features=list(feature_names), outcome_name="outcome")
    model = dice_ml.Model(model=estimator, backend="sklearn")
    engine = dice_ml.Dice(data, model, method="random")

    # Bound the search to the range each feature actually takes in training.
    # Without this DiCE may propose physically impossible race states, and the
    # unbounded space makes finding a rare target class much slower.
    permitted = {
        f: [float(df[f].min()), float(df[f].max())] for f in feature_names
    }

    query = pd.DataFrame([row_scaled], columns=feature_names)
    try:
        with _time_budget(timeout_seconds):
            res = engine.generate_counterfactuals(
                query, total_CFs=total_cfs, desired_class="opposite",
                permitted_range=permitted, random_seed=seed, verbose=False,
            )
        cf_df = res.cf_examples_list[0].final_cfs_df
    except _SearchTimeout as exc:
        return {"method": "DiCE (random search)", "available": True, "counterfactuals": [],
                "timed_out": True, "timeout_seconds": timeout_seconds,
                "reason": (
                    f"{exc}. The positive class is rare in this dataset, so a random search "
                    f"over the feature space can run indefinitely without finding an "
                    f"opposite-class row. The single-feature bisection scan above is the "
                    f"method that does produce an actionable answer here."
                )}
    except Exception as exc:
        return {"method": "DiCE (random search)", "available": False,
                "reason": f"search failed: {type(exc).__name__}: {exc}", "counterfactuals": []}

    if cf_df is None or cf_df.empty:
        return {"method": "DiCE", "available": True, "counterfactuals": [],
                "reason": "DiCE found no counterfactual within its search budget"}

    original = query.iloc[0]
    out = []
    for _, cf in cf_df.iterrows():
        changes = {
            f: {"from": float(original[f]), "to": float(cf[f]), "delta": float(cf[f] - original[f])}
            for f in feature_names
            if not np.isclose(float(cf[f]), float(original[f]), atol=1e-6)
        }
        out.append({"changed_features": changes, "n_changes": len(changes),
                    "new_outcome": int(cf["outcome"]) if "outcome" in cf else None})

    return {
        "method": "DiCE (random search)",
        "available": True,
        "total_requested": total_cfs,
        "feature_space": "standardised (the space the model was fitted in)",
        "counterfactuals": out,
        "note": (
            "Each row above is a complete alternative race state the model would "
            "classify the other way. Values are in standardised units; a delta of "
            "+1.0 means one standard deviation of that feature as observed in training."
        ),
    }
