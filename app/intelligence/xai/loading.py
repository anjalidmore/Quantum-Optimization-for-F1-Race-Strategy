"""
app.intelligence.xai.loading
============================

Assembles everything Task 8 needs to explain, for one target: the data, the
split, Task 7's deep network, and **Task 6's actual persisted pipeline**.

This is the key difference from the ``task-mode`` counterpart. There each
practical is self-contained, so a classical champion is re-fit locally. Here
Task 6 has already trained, selected and persisted its pipelines to
``artifacts/models/``, and ``ModelCache`` already knows how to load the
selected-best one. Task 8 explains *that* model - the one the API serves and
the dashboard displays - rather than a private re-fit that might differ.

Both model families are wrapped in a uniform ``predict(rows) -> 1-D array``
callable so every explainer downstream is model-agnostic by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.paths import DL_MODELS_DIR  # noqa: E402
from app.intelligence.dl import persistence as dl_persistence  # noqa: E402
from app.intelligence.dl import training as dl_training  # noqa: E402
from app.intelligence.ml.data_contract import build_task_frame, load_and_validate  # noqa: E402
from app.intelligence.ml.splits import chronological_holdout  # noqa: E402
from app.services.model_cache import ModelUnavailableError, get_model_cache  # noqa: E402

TARGET_TASKS = {
    "target_laptime": "regression",
    "target_pit_next_lap": "classification",
}


class ExplainerUnavailableError(RuntimeError):
    """Raised when a model Task 8 needs to explain has not been trained yet.
    Task 8 never substitutes a stand-in model - it says what is missing."""


@dataclass
class ExplainableTarget:
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
    classical_pipeline: object
    classical_X_train_transformed: np.ndarray
    classical_X_test_transformed: np.ndarray
    transformed_feature_names: list[str]
    source_dataset: str

    @property
    def n_identity(self) -> int:
        return len(self.identity_features)


def load_target(target: str) -> ExplainableTarget:
    if target not in TARGET_TASKS:
        raise ValueError(f"Unknown target {target!r}; expected one of {list(TARGET_TASKS)}")
    task = TARGET_TASKS[target]

    dataset = load_and_validate()
    contract = dataset.contract
    frame, features = build_task_frame(dataset, target)

    binary = set(contract.binary_features_no_scaling_needed)
    mask = np.array([f not in binary for f in features], dtype=bool)
    X = frame[features].to_numpy(dtype="float32")
    y = frame[target].to_numpy(dtype="float32")

    holdout = chronological_holdout(frame, test_fraction=0.2)
    X_tr, y_tr = X[holdout.dev_index], y[holdout.dev_index]
    X_te, y_te = X[holdout.test_index], y[holdout.test_index]
    laps = frame.loc[holdout.test_index, "LapNumber"].to_numpy()

    # --- Task 7 deep network ------------------------------------------------
    try:
        model, scaler, y_scaler, spec = dl_persistence.load(target, DL_MODELS_DIR)
    except FileNotFoundError as exc:
        raise ExplainerUnavailableError(
            f"No Task 7 deep model for {target!r} at {DL_MODELS_DIR}. "
            f"Run `python scripts/build_all.py` to train it."
        ) from exc
    if spec["features"] != features:
        raise ExplainerUnavailableError(
            f"The saved deep model for {target!r} expects a different feature order than "
            f"the current Task 5 contract provides. Retrain Task 7."
        )

    def dnn_predict(rows: np.ndarray) -> np.ndarray:
        return dl_training.predict(
            model, scaler, np.asarray(rows, dtype="float32"), mask, y_scaler)

    # --- Task 6's persisted, selected-best pipeline --------------------------
    cache = get_model_cache()
    try:
        classical_name, pipeline = cache.get_pipeline(target)
    except ModelUnavailableError as exc:
        raise ExplainerUnavailableError(
            f"No Task 6 pipeline for {target!r}. Run `python scripts/build_all.py` first."
        ) from exc

    frame_features = pd.DataFrame(X, columns=features)

    def classical_predict(rows: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(np.asarray(rows, dtype="float32"), columns=features)
        if task == "classification" and hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(df)[:, 1]
        return pipeline.predict(df)

    # SHAP's TreeExplainer needs the estimator and the data *after* the
    # pipeline's preprocessing, not the raw frame - the tree was fitted in
    # transformed space and attributions must be computed there.
    pre = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    X_tr_t = np.asarray(pre.transform(frame_features.iloc[holdout.dev_index]), dtype=float)
    X_te_t = np.asarray(pre.transform(frame_features.iloc[holdout.test_index]), dtype=float)
    tnames = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]

    return ExplainableTarget(
        target=target, task=task, features=features,
        identity_features=[f for f in features if f.startswith(("driver_", "team_"))],
        numeric_mask=mask,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        test_lap_numbers=laps,
        dnn_predict=dnn_predict,
        classical_predict=classical_predict,
        classical_name=classical_name,
        classical_estimator=estimator,
        classical_pipeline=pipeline,
        classical_X_train_transformed=X_tr_t,
        classical_X_test_transformed=X_te_t,
        transformed_feature_names=tnames,
        source_dataset=contract.source_dataset,
    )


def pick_representative_rows(t: ExplainableTarget) -> dict[str, int]:
    """Test-set rows worth explaining in detail, de-duplicated.

    Labels describe *what was actually found*: on a holdout where the model
    never predicts above 0.5, calling its highest-probability row "clear pit
    now" would misrepresent it, so the label says "highest pit probability".
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
    """The test row closest to the decision boundary - the best starting point
    for a counterfactual search, because a flip is most likely reachable there."""
    pred = np.asarray(t.dnn_predict(t.X_test)).ravel()
    threshold = 0.5 if t.task == "classification" else float(np.median(t.y_train))
    return int(np.argmin(np.abs(pred - threshold)))
