"""
f1dl.tuning
===========

A small, bounded hyperparameter search evaluated on Task 6's expanding-window
lap-forward folds.

Why not K-fold or a random search? The Task 5 contract is explicit:
"Expanding-window lap-forward split; keep whole laps in one fold. Do not use
random K-fold - this is a time-ordered panel." A random split would let the
network see lap 40 while being scored on lap 20, which is not a situation any
deployed race-strategy model will ever be in. The grid is deliberately small
and fully enumerated (not sampled) so the search is deterministic and the
whole space explored can be printed in the report.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import evaluation, training


@dataclass(frozen=True)
class SearchSpace:
    hidden_units: tuple[tuple[int, ...], ...]
    dropout: tuple[float, ...]
    learning_rate: tuple[float, ...]
    batch_size: tuple[int, ...]

    def combinations(self) -> list[dict]:
        combos = itertools.product(self.hidden_units, self.dropout, self.learning_rate, self.batch_size)
        return [
            {"hidden_units": h, "dropout": d, "learning_rate": lr, "batch_size": b}
            for h, d, lr, b in combos
        ]

    def to_metadata(self) -> dict:
        return {
            "hidden_units": [list(h) for h in self.hidden_units],
            "dropout": list(self.dropout),
            "learning_rate": list(self.learning_rate),
            "batch_size": list(self.batch_size),
            "total_combinations": len(self.combinations()),
        }


REGRESSION_SPACE = SearchSpace(
    hidden_units=((32, 16), (64, 32)),
    dropout=(0.1, 0.3),
    learning_rate=(1e-3, 3e-4),
    batch_size=(32,),
)

CLASSIFICATION_SPACE = SearchSpace(
    hidden_units=((16, 8), (32, 16)),
    dropout=(0.2, 0.4),
    learning_rate=(1e-3, 3e-4),
    batch_size=(32,),
)


@dataclass
class TrialResult:
    params: dict
    fold_metrics: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    mean_epochs: float = 0.0
    # Pooled out-of-fold predictions (classification only): every value was
    # produced by a network that had not seen that row, so a threshold tuned on
    # them is honest in the same way a CV score is.
    oof_y_true: list = field(default_factory=list)
    oof_y_proba: list = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "params": {**self.params, "hidden_units": list(self.params["hidden_units"])},
            "cv_summary": self.summary,
            "mean_epochs_to_best": self.mean_epochs,
            "per_fold": self.fold_metrics,
        }


def _score_regression(y_true, y_pred) -> dict:
    return evaluation.regression_metrics(y_true, y_pred)


def _score_classification(y_true, y_proba) -> dict:
    y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    return evaluation.classification_metrics(y_true, y_pred, y_proba=y_proba)


def search(
    build_fn,
    X: np.ndarray,
    y: np.ndarray,
    folds: list,
    numeric_mask: np.ndarray,
    space: SearchSpace,
    *,
    task: str,
    class_weight: dict | None = None,
    scale_target: bool = False,
    max_epochs: int = 200,
    patience: int = 20,
    log=None,
) -> tuple[dict, list[TrialResult]]:
    """Evaluate every combination in ``space`` across ``folds``.

    Returns ``(best_params, all_trials)``. Selection is by mean CV MAE for
    regression and mean CV ROC-AUC for classification - the same primary
    metrics Task 6 selects on, so the two tasks' winners are chosen by the
    same rule.
    """
    metric_keys = (
        ["mae", "rmse", "r2"] if task == "regression" else
        ["roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall"]
    )
    trials: list[TrialResult] = []

    for i, params in enumerate(space.combinations(), start=1):
        fold_metrics: list[dict] = []
        epochs: list[int] = []
        oof_true: list = []
        oof_proba: list = []
        for fold in folds:
            tr, va = fold.train_index, fold.val_index
            fit = training.fit_fold(
                build_fn,
                X[tr], y[tr], X[va], y[va],
                numeric_mask,
                hidden_units=params["hidden_units"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                max_epochs=max_epochs,
                patience=patience,
                class_weight=class_weight,
                scale_target=scale_target,
            )
            pred = training.predict(fit.model, fit.scaler, X[va], numeric_mask, fit.y_scaler)
            m = _score_regression(y[va], pred) if task == "regression" else _score_classification(y[va], pred)
            m["fold_id"] = fold.fold_id
            fold_metrics.append(m)
            epochs.append(fit.best_epoch)

            if task == "classification":
                oof_true.extend(list(y[va]))
                oof_proba.extend(list(np.asarray(pred).ravel()))

        summary = evaluation.aggregate_metrics(fold_metrics, metric_keys)
        trial = TrialResult(
            params=params,
            fold_metrics=fold_metrics,
            summary=summary,
            mean_epochs=float(np.mean(epochs)) if epochs else 0.0,
            oof_y_true=oof_true,
            oof_y_proba=oof_proba,
        )
        trials.append(trial)
        if log:
            primary = "mae" if task == "regression" else "pr_auc"
            val = summary.get(primary, {}).get("mean")
            log.info(
                "  trial %d/%d  %s  CV %s=%s",
                i, len(space.combinations()),
                {**params, "hidden_units": list(params["hidden_units"])},
                primary,
                f"{val:.4f}" if val is not None else "undefined",
            )

    best = _pick_best(trials, task)
    return best.params, trials


def _pick_best(trials: list[TrialResult], task: str) -> TrialResult:
    if task == "regression":
        scored = [(t.summary["mae"]["mean"], t) for t in trials if t.summary["mae"]["mean"] is not None]
        if not scored:
            raise RuntimeError("No regression trial produced a defined CV MAE.")
        return min(scored, key=lambda p: p[0])[1]

    # PR-AUC first: ROC-AUC is uninformative at this prevalence.
    scored = [(t.summary["pr_auc"]["mean"], t) for t in trials if t.summary["pr_auc"]["mean"] is not None]
    if scored:
        return max(scored, key=lambda p: p[0])[1]
    scored = [(t.summary["roc_auc"]["mean"], t) for t in trials if t.summary["roc_auc"]["mean"] is not None]
    if scored:
        return max(scored, key=lambda p: p[0])[1]
    # Every fold's ROC-AUC was undefined (no positive class in any validation
    # block). Fall back to F1 rather than silently returning an arbitrary
    # trial, and let the report say the fallback was used.
    scored = [(t.summary["f1"]["mean"], t) for t in trials if t.summary["f1"]["mean"] is not None]
    if not scored:
        raise RuntimeError("No classification trial produced a defined CV metric.")
    return max(scored, key=lambda p: p[0])[1]
