"""
f1dl.threshold
==============

Choosing the probability cut-off at which "pit" is actually recommended.

**Why this module exists.** A binary classifier outputs a probability; turning
that into a decision needs a threshold. ``predict()`` uses 0.5, which is not a
considered choice — it is a default that happens to be optimal only when the
classes are balanced *and* the two error types cost the same. Neither holds
here: pit events are 4.8% of laps, and missing a pit window is not the same
kind of mistake as pitting a lap early.

The consequence was measurable. Every Task 6 and Task 7 pit classifier scored a
respectable ROC-AUC — meaning it *ranks* laps by pit risk correctly — while
being useless as a decision rule at 0.5. Two illustrative baselines:

* the Task 7 network: ROC-AUC 0.92, and F1 / precision / recall all exactly
  **0.0**, because at 0.5 it predicted "never pit" for all 180 test laps and
  rode the class imbalance to 99.4% accuracy;
* Task 6's ``random_forest``: recall 1.0 but precision 0.04 — it fired on
  25 laps to catch the one real pit event.

Both are threshold problems, not ranking problems, and no amount of retraining
fixes them.

**How the threshold is chosen.** On **pooled out-of-fold predictions** from the
expanding-window CV folds — never on the test set, and never on data a fold's
model was fitted to. Pooling matters here: each validation fold holds only 6-11
positives, so a per-fold threshold averaged afterwards is dominated by noise,
whereas the pooled set has ~36 and gives a stable curve.

Two objectives are supported and both are reported:

``f1``
    Maximises the harmonic mean of precision and recall — the standard choice
    when you have no explicit cost model.

``expected_cost``
    Minimises ``fn_cost * false_negatives + fp_cost * false_positives``. This is
    the one that actually reflects racing: missing the pit window costs a
    driver far more than an unnecessary early stop, so ``fn_cost`` defaults to
    5x ``fp_cost``. The ratio is an assumption, stated here rather than buried,
    and exposed so it can be argued with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A missed pit window (losing track position, or running a tyre past its cliff)
# is judged five times as costly as an unnecessary early stop. This is a domain
# assumption, not a measurement — it is surfaced in the report so a reader can
# disagree with it explicitly.
DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0

# 0.5 is what sklearn's predict() uses. Recorded so before/after comparisons
# always state what the baseline actually was.
DEFAULT_THRESHOLD = 0.5


@dataclass
class ThresholdChoice:
    """The chosen cut-off and everything needed to justify it."""

    threshold: float
    objective: str
    n_samples: int
    n_positive: int
    # Metrics at the chosen threshold, on the data it was chosen from.
    f1: float
    precision: float
    recall: float
    expected_cost: float
    # The same metrics at 0.5, so the improvement is always visible.
    baseline: dict = field(default_factory=dict)
    curve: list[dict] = field(default_factory=list)
    note: str = ""

    def to_metadata(self) -> dict:
        return {
            "threshold": round(float(self.threshold), 4),
            "objective": self.objective,
            "chosen_on": "pooled out-of-fold CV predictions",
            "n_samples": int(self.n_samples),
            "n_positive": int(self.n_positive),
            "at_threshold": {
                "f1": round(float(self.f1), 4),
                "precision": round(float(self.precision), 4),
                "recall": round(float(self.recall), 4),
                "expected_cost": round(float(self.expected_cost), 2),
            },
            "at_default_0.5": self.baseline,
            "note": self.note,
        }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def metrics_at(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> dict:
    """Precision / recall / F1 / expected cost at one cut-off."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tp, fp, fn, _ = _confusion(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "expected_cost": float(fn_cost * fn + fp_cost * fp),
        "tp": tp, "fp": fp, "fn": fn,
        "n_predicted_positive": int(tp + fp),
    }


def tune_threshold(
    y_true,
    y_proba,
    objective: str = "f1",
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    n_candidates: int = 199,
) -> ThresholdChoice:
    """Pick the cut-off optimising ``objective`` on ``(y_true, y_proba)``.

    ``y_proba`` must be **out-of-fold** predictions. Tuning a threshold on data
    the model was fitted to gives an optimistic cut-off that will not hold, in
    exactly the way tuning any other hyperparameter on training data does.

    If the input has no positive examples the threshold is left at 0.5 and the
    reason is recorded — a cut-off cannot be chosen from data containing none of
    the class it is meant to detect, and silently returning some number would be
    a fabricated result.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_pos = int(y_true.sum())

    baseline = metrics_at(y_true, y_proba, DEFAULT_THRESHOLD, fn_cost, fp_cost)

    if n_pos == 0 or n_pos == len(y_true):
        return ThresholdChoice(
            threshold=DEFAULT_THRESHOLD, objective=objective,
            n_samples=len(y_true), n_positive=n_pos,
            f1=baseline["f1"], precision=baseline["precision"],
            recall=baseline["recall"], expected_cost=baseline["expected_cost"],
            baseline={k: round(v, 4) for k, v in baseline.items() if isinstance(v, float)},
            note=(
                f"Only one class present in the tuning data ({n_pos} positive of "
                f"{len(y_true)}); no threshold is learnable, so the default 0.5 is kept."
            ),
        )

    # Candidate cut-offs spanning the observed probability range. Using the
    # observed range rather than [0, 1] matters when a model's probabilities are
    # compressed into a narrow band, which is common under class weighting.
    lo, hi = float(np.min(y_proba)), float(np.max(y_proba))
    if hi <= lo:
        candidates = np.array([DEFAULT_THRESHOLD])
    else:
        candidates = np.linspace(lo, hi, n_candidates)

    curve = [metrics_at(y_true, y_proba, t, fn_cost, fp_cost) for t in candidates]

    if objective == "f1":
        best = max(curve, key=lambda r: (r["f1"], r["recall"]))
    elif objective == "expected_cost":
        best = min(curve, key=lambda r: (r["expected_cost"], -r["f1"]))
    else:
        raise ValueError(f"Unknown objective {objective!r}; expected 'f1' or 'expected_cost'")

    improvement = best["f1"] - baseline["f1"]
    return ThresholdChoice(
        threshold=best["threshold"], objective=objective,
        n_samples=len(y_true), n_positive=n_pos,
        f1=best["f1"], precision=best["precision"],
        recall=best["recall"], expected_cost=best["expected_cost"],
        baseline={k: round(v, 4) for k, v in baseline.items() if isinstance(v, float)},
        curve=[{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in curve],
        note=(
            f"Chosen on {len(y_true)} pooled out-of-fold predictions containing {n_pos} "
            f"positives. F1 {baseline['f1']:.4f} at the default 0.5 -> {best['f1']:.4f} at "
            f"{best['threshold']:.4f} ({improvement:+.4f})."
        ),
    )


def apply(y_proba, threshold: float) -> np.ndarray:
    """Turn probabilities into decisions at ``threshold``."""
    return (np.asarray(y_proba, dtype=float) >= float(threshold)).astype(int)
