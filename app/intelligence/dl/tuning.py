"""
app.intelligence.dl.tuning
==========================

Hyperparameter selection for the Task 7 networks, scored on Task 6's
expanding-window lap-forward folds.

**Why these folds, not K-fold.** The Task 5 contract is explicit:
"Expanding-window lap-forward split; keep whole laps in one fold. Do not use
random K-fold - this is a time-ordered panel." Every fold trains on earlier
laps and validates on the block of laps immediately after, so the network is
never scored on a race state that precedes the ones it learned from. Task 7
imports ``app.intelligence.ml.splits`` and ``.evaluation`` directly rather than
copying them, so the deep and classical numbers come from the same code.

**Why one-factor-at-a-time.** The specification's candidate values (3 depths ×
3 learning rates × 3 batch sizes × 4 dropouts × 2 optimizers × 2 L2 settings ×
2 loss/weighting options) make 2,592 combinations; at four folds each that is
over ten thousand network fits on a laptop CPU. Instead each *stage* varies one
hyperparameter while every other one holds the best value found so far. Every
listed value is still tried, every experiment is recorded, and the test set is
never consulted — selection uses the cross-validated metric only.

Selection metric: mean CV **MAE** for lap time (lower is better) and mean CV
**PR-AUC** for the pit decision (higher is better). PR-AUC rather than ROC-AUC
because pit events are 4.8% of laps, and at that prevalence ROC-AUC stays high
for a model that never fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.intelligence.dl import training
from app.intelligence.ml import evaluation

BASE_PARAMS = {
    # The specification's suggested starting network. It is a starting point,
    # not a conclusion: the architecture stage below replaces it if a smaller
    # network validates better, which on ~800 development rows is expected.
    "hidden_units": (128, 64, 32),
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "optimizer": "adam",
    "l2": 1e-4,
}

REGRESSION_STAGES: list[tuple[str, list]] = [
    ("hidden_units", [(128, 64, 32), (64, 32), (32, 16)]),
    ("learning_rate", [1e-3, 5e-4, 1e-4]),
    ("batch_size", [32, 16, 64]),
    ("dropout", [0.3, 0.2, 0.4, 0.5]),
    ("optimizer", ["adam", "rmsprop"]),
    ("l2", [1e-4, 1e-3]),
    ("loss", ["mse", "huber"]),
]

CLASSIFICATION_STAGES: list[tuple[str, list]] = [
    # Class weighting is investigated first and on validation data, rather than
    # applied by default: with 4.8% positives it may help recall, or it may just
    # inflate every probability. Only the folds can say which.
    ("class_weighted", [False, True]),
    ("hidden_units", [(128, 64, 32), (32, 16), (16, 8)]),
    ("learning_rate", [1e-3, 5e-4, 1e-4]),
    ("batch_size", [32, 16, 64]),
    ("dropout", [0.3, 0.2, 0.4, 0.5]),
    ("optimizer", ["adam", "rmsprop"]),
    ("l2", [1e-4, 1e-3]),
]

STAGE_LABEL = {
    "hidden_units": "architecture (layers / neurons)",
    "learning_rate": "learning rate",
    "batch_size": "batch size",
    "dropout": "dropout",
    "optimizer": "optimizer",
    "l2": "L2 regularisation",
    "loss": "loss function",
    "class_weighted": "class weighting",
}


def stages_for(task: str, quick: bool = False) -> list[tuple[str, list]]:
    stages = REGRESSION_STAGES if task == "regression" else CLASSIFICATION_STAGES
    if quick:
        # A smoke run: two stages, two values each. Never used for committed results.
        return [(name, values[:2]) for name, values in stages[:2]]
    return stages


def base_params(task: str) -> dict:
    extra = {"loss": "mse"} if task == "regression" else {"class_weighted": False}
    return {**BASE_PARAMS, **extra}


def _key(params: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()))


@dataclass
class TrialResult:
    experiment: str
    stage: str
    params: dict
    fold_metrics: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    mean_epochs: float = 0.0
    mean_best_epoch: float = 0.0
    parameters: int = 0
    # Pooled out-of-fold predictions: every value came from a network that had
    # not seen that row, so a decision threshold tuned on them is honest in the
    # same way a CV score is.
    oof_y_true: list = field(default_factory=list)
    oof_y_pred: list = field(default_factory=list)

    @property
    def oof_y_proba(self) -> list:  # name used by the classification caller
        return self.oof_y_pred

    def to_metadata(self) -> dict:
        return {
            "experiment": self.experiment,
            "varied": STAGE_LABEL.get(self.stage, self.stage),
            "params": {**self.params, "hidden_units": list(self.params["hidden_units"])},
            "parameters": self.parameters,
            "cv_summary": self.summary,
            "mean_epochs_run": self.mean_epochs,
            "mean_best_epoch": self.mean_best_epoch,
            "per_fold": self.fold_metrics,
        }


def _score(task: str, y_true, pred) -> dict:
    if task == "regression":
        return evaluation.regression_metrics(y_true, pred)
    y_hat = (np.asarray(pred) >= 0.5).astype(int)
    return evaluation.classification_metrics(y_true, y_hat, y_proba=pred)


def primary_metric(task: str) -> str:
    return "mae" if task == "regression" else "pr_auc"


def _rank_key(task: str, trial: TrialResult):
    """Sort key where larger is better. Undefined primaries rank last."""
    m = trial.summary.get(primary_metric(task), {}).get("mean")
    if m is None:
        return (-np.inf,)
    if task == "regression":
        return (-m, -(trial.summary.get("rmse", {}).get("mean") or np.inf))
    return (m, trial.summary.get("roc_auc", {}).get("mean") or -np.inf)


def evaluate_params(build_fn, X, y, folds, mask, task: str, params: dict, *,
                    scale_target: bool, max_epochs: int, patience: int) -> TrialResult:
    trial = TrialResult(experiment="", stage="", params=params)
    epochs, best_epochs = [], []
    for fold in folds:
        tr, va = fold.train_index, fold.val_index
        cw = None
        if task == "classification" and params.get("class_weighted"):
            # Balanced weights from THIS fold's training labels only.
            cw = training.balanced_class_weight(y[tr])
        fit = training.fit_fold(
            build_fn, X[tr], y[tr], X[va], y[va], mask,
            hidden_units=params["hidden_units"], dropout=params["dropout"],
            learning_rate=params["learning_rate"], batch_size=params["batch_size"],
            optimizer=params["optimizer"], l2=params["l2"],
            loss=params.get("loss"), class_weight=cw, scale_target=scale_target,
            max_epochs=max_epochs, patience=patience,
        )
        pred = training.predict(fit.model, fit.scaler, X[va], mask, fit.y_scaler)
        m = _score(task, y[va], pred)
        m.update(fold_id=fold.fold_id, epochs_run=fit.epochs_run, best_epoch=fit.best_epoch)
        trial.fold_metrics.append(m)
        trial.oof_y_true.extend(np.asarray(y[va]).ravel().tolist())
        trial.oof_y_pred.extend(np.asarray(pred).ravel().tolist())
        epochs.append(fit.epochs_run)
        best_epochs.append(fit.best_epoch)
        trial.parameters = int(fit.model.count_params())

    keys = (["mae", "rmse", "r2"] if task == "regression"
            else ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"])
    trial.summary = evaluation.aggregate_metrics(trial.fold_metrics, keys)
    trial.mean_epochs = float(np.mean(epochs))
    trial.mean_best_epoch = float(np.mean(best_epochs))
    return trial


def search(build_fn, X, y, folds, mask, *, task: str, scale_target: bool = False,
           max_epochs: int = 200, patience: int = 20, quick: bool = False, log=None
           ) -> tuple[dict, list[TrialResult]]:
    """One-factor-at-a-time search over the expanding-window folds.

    Returns ``(best_params, trials)``; ``trials`` lists every distinct
    configuration trained, in order, each labelled with the stage that
    introduced it.
    """
    best = base_params(task)
    cache: dict[tuple, TrialResult] = {}
    trials: list[TrialResult] = []
    metric = primary_metric(task)

    for stage, values in stages_for(task, quick):
        contenders: list[TrialResult] = []
        for v in values:
            params = {**best, stage: v}
            k = _key(params)
            if k not in cache:
                trial = evaluate_params(build_fn, X, y, folds, mask, task, params,
                                        scale_target=scale_target, max_epochs=max_epochs,
                                        patience=patience)
                trial.experiment = f"E{len(trials) + 1:02d}"
                trial.stage = stage
                cache[k] = trial
                trials.append(trial)
                if log:
                    s = trial.summary.get(metric, {})
                    log.info("  %s %-32s %-28s CV %s %s (± %s) | %d params | %.0f epochs",
                             trial.experiment, STAGE_LABEL[stage], f"{stage}={_fmt(v)}", metric,
                             "undefined" if s.get("mean") is None else f"{s['mean']:.4f}",
                             "-" if s.get("std") is None else f"{s['std']:.4f}",
                             trial.parameters, trial.mean_epochs)
            contenders.append(cache[k])
        winner = max(contenders, key=lambda tr: _rank_key(task, tr))
        best = dict(winner.params)
        if log:
            log.info("  -> keep %s=%s", stage, _fmt(best[stage]))
    return best, trials


def _fmt(v) -> str:
    return str(list(v)) if isinstance(v, tuple) else str(v)
