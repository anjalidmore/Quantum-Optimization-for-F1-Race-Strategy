"""
f1xai.loading
=============

Assembles everything Task 8 needs to explain: the data, the split, the Task 7
deep network, and a classical champion trained on the identical split.

Task 8's remit is to explain *both* model families. On ``task-mode`` each
practical is self-contained, so the classical champion is re-fit here from
practical06's ``f1dl.baselines`` specs under the same seed and the same
fold-local scaling - deterministic, and identical to what practical06
compared against. (On ``proj-mode`` the unified app loads Task 6's persisted
``.joblib`` pipelines instead.)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402


def _add_practical06_to_path() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        p06 = candidate / "practical06"
        if p06.is_dir():
            sys.path.insert(0, str(p06 / "src"))
            return p06
    raise FileNotFoundError(
        "Could not locate 'practical06'. Task 8 explains Task 7's models, so "
        "practical06 must be a sibling directory and must have been run."
    )


PRACTICAL06 = _add_practical06_to_path()

from f1dl import baselines, contract, splits, training  # noqa: E402
from f1dl import persistence as dl_persistence  # noqa: E402


@dataclass
class ExplainableTarget:
    """Everything needed to explain one target, with both model families."""

    target: str
    task: str
    features: list[str]
    identity_features: list[str]
    numeric_mask: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_lap_numbers: np.ndarray
    dnn_predict: callable
    classical_predict: callable
    classical_name: str
    classical_estimator: object
    classical_X_train_scaled: np.ndarray
    classical_X_test_scaled: np.ndarray
    source_dataset: str

    @property
    def n_identity(self) -> int:
        return len(self.identity_features)


CLASSICAL_CHAMPION = {
    "target_laptime": "random_forest",
    "target_pit_next_lap": "random_forest",
}


def load_target(target: str, models_dir: Path | None = None) -> ExplainableTarget:
    if models_dir is None:
        models_dir = PRACTICAL06 / "outputs" / "models"

    dataset = contract.load_and_validate()
    c = dataset.contract
    task = "regression" if target == "target_laptime" else "classification"

    frame, features = contract.build_task_frame(dataset, target)
    binary = set(c.binary_features_no_scaling_needed)
    mask = np.array([f not in binary for f in features], dtype=bool)
    X = frame[features].to_numpy(dtype="float32")
    y = frame[target].to_numpy(dtype="float32")

    holdout = splits.chronological_holdout(frame, test_fraction=0.2)
    X_tr, y_tr = X[holdout.dev_index], y[holdout.dev_index]
    X_te, y_te = X[holdout.test_index], y[holdout.test_index]
    laps = frame.loc[holdout.test_index, "LapNumber"].to_numpy()

    # --- Task 7 deep network -------------------------------------------------
    model, scaler, y_scaler, spec = dl_persistence.load(target, models_dir)
    if spec["features"] != features:
        raise RuntimeError(
            f"Saved DNN for {target!r} expects a different feature order than the "
            f"contract provides. Re-run practical06."
        )

    def dnn_predict(rows: np.ndarray) -> np.ndarray:
        return training.predict(model, scaler, np.asarray(rows, dtype="float32"), mask, y_scaler)

    # --- Classical champion, identical split, identical scaling --------------
    specs = baselines.regression_baselines() if task == "regression" else baselines.classification_baselines()
    name = CLASSICAL_CHAMPION[target]
    est = specs[name]
    Xtr_s, Xte_s, _ = training.scale_fit_transform(X_tr, X_te, mask)
    est.fit(Xtr_s, y_tr)

    cls_scaler = training.scale_fit_transform(X_tr, X_tr, mask)[2]

    def classical_predict(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype="float32").copy()
        if mask.any():
            rows[:, mask] = cls_scaler.transform(np.asarray(rows)[:, mask])
        if task == "classification" and hasattr(est, "predict_proba"):
            return est.predict_proba(rows)[:, 1]
        return est.predict(rows)

    return ExplainableTarget(
        target=target, task=task, features=features,
        identity_features=c.identity_features(target),
        numeric_mask=mask,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        test_lap_numbers=laps,
        dnn_predict=dnn_predict,
        classical_predict=classical_predict,
        classical_name=name, classical_estimator=est,
        classical_X_train_scaled=Xtr_s, classical_X_test_scaled=Xte_s,
        source_dataset=c.source_dataset,
    )


def pick_representative_rows(t: ExplainableTarget) -> dict[str, int]:
    """Choose test-set rows worth explaining in detail.

    For the pit classifier: the lowest pit probability, the row closest to the
    0.5 decision boundary, and the highest pit probability - the three cases a
    race engineer cares about. Labels say *what was actually found* rather than
    promising a "clear pit now" case that may not exist: on a holdout with no
    pit events the model may never predict above 0.5, and calling its
    highest-probability row "clear pit now" would misrepresent it.

    Rows are de-duplicated. If the highest-probability row is also the row
    closest to the boundary (which happens exactly when no prediction reaches
    0.5), only one entry survives rather than the same row being explained
    twice under two names.
    """
    pred = np.asarray(t.dnn_predict(t.X_test)).ravel()
    order = np.argsort(pred)

    if t.task == "classification":
        candidates = [
            ("lowest_pit_probability", int(order[0])),
            ("closest_to_decision_boundary", int(np.argmin(np.abs(pred - 0.5)))),
            ("highest_pit_probability", int(order[-1])),
        ]
    else:
        candidates = [
            ("fastest_predicted_lap", int(order[0])),
            ("median_predicted_lap", int(order[len(order) // 2])),
            ("slowest_predicted_lap", int(order[-1])),
        ]

    out: dict[str, int] = {}
    seen: set[int] = set()
    for label, idx in candidates:
        if idx not in seen:
            out[label] = idx
            seen.add(idx)
    return out


def boundary_row_index(t: ExplainableTarget) -> int:
    """The test row whose prediction sits closest to the decision boundary -
    the best starting point for a counterfactual search, because it is where a
    flip is most likely to be reachable."""
    pred = np.asarray(t.dnn_predict(t.X_test)).ravel()
    threshold = 0.5 if t.task == "classification" else float(np.median(t.y_train))
    return int(np.argmin(np.abs(pred - threshold)))
