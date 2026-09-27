"""
app.intelligence.qml.data
=========================

Turning the Task 5 feature matrix into circuit inputs, without changing the
experiment Task 6 and Task 7 already ran.

Three things matter here, and all three are about not cheating:

* **The same data and the same splits.** This module imports
  ``app.intelligence.ml.data_contract`` and ``app.intelligence.ml.splits``
  rather than copying them, so the quantum models are scored on exactly the
  laps the classical and deep models were scored on.
* **Everything is fitted on training rows only.** The standardiser, the
  [0, π] angle map and the PCA are all fitted inside the fold (or on the
  development set for the final model) and then applied to the held-out rows.
  Fitting any of them on the full matrix would leak the test laps into
  training through the transform.
* **Angle encoding needs a bounded range.** ``AngleEmbedding`` turns each
  feature into a rotation angle, so the features are mapped to [0, π]. Angles
  wrap, and a value outside the range would alias onto a different rotation.
  Test rows that fall outside the training range are clipped, and the clipping
  is counted and reported rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.intelligence.ml.data_contract import build_task_frame, load_and_validate
from app.intelligence.ml.splits import chronological_holdout, expanding_window_folds

# Four qubits keeps every circuit in this module small enough to simulate in
# milliseconds, which is what makes a real hyperparameter search affordable
# here. The cost is real and is reported: 45 lap-time features compressed into
# 4 PCA components lose variance, which is exactly why the classical baseline
# is also trained on these same 4 components.
N_QUBITS = 4
ANGLE_RANGE = (0.0, np.pi)


@dataclass
class QuantumTaskData:
    """One target, split and encoded, ready for a circuit."""

    target: str
    task: str                       # "classification" | "regression"
    feature_names: list[str]
    X_dev: np.ndarray               # development rows, encoded (angles)
    y_dev: np.ndarray
    X_test: np.ndarray              # chronological holdout, encoded with the dev transforms
    y_test: np.ndarray
    dev_index: np.ndarray
    test_index: np.ndarray
    folds: list                     # Task 6's expanding-window folds, positions within dev
    holdout: dict
    reduction: dict                 # which method reduced the features, and what it cost
    n_qubits: int = N_QUBITS
    y_dev_raw: np.ndarray | None = None      # regression: unstandardised target
    y_test_raw: np.ndarray | None = None
    target_mean: float | None = None
    target_std: float | None = None
    notes: list[str] = field(default_factory=list)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        """Standardised predictions back to seconds (regression only)."""
        if self.target_std is None:
            return np.asarray(y)
        return np.asarray(y) * self.target_std + self.target_mean


class AngleEncoder:
    """Standardise → PCA (if needed) → map to [0, π]. Fitted on training rows only.

    Kept as one small class because the three steps have to be fitted and
    applied as a unit: a PCA fitted on unstandardised columns would be driven by
    whichever feature has the largest units, and an angle map fitted after the
    test rows arrive would leak their range.
    """

    def __init__(self, n_qubits: int = N_QUBITS):
        self.n_qubits = n_qubits
        self.standardiser = StandardScaler()
        self.pca: PCA | None = None
        self.angles = MinMaxScaler(feature_range=ANGLE_RANGE)
        self.method: str = "unfitted"
        self.explained_variance: float | None = None
        self.n_clipped_: int = 0

    def fit(self, X: np.ndarray) -> "AngleEncoder":
        Z = self.standardiser.fit_transform(np.asarray(X, dtype=float))
        if Z.shape[1] > self.n_qubits:
            self.pca = PCA(n_components=self.n_qubits, random_state=42).fit(Z)
            Z = self.pca.transform(Z)
            self.method = f"PCA to {self.n_qubits} components"
            self.explained_variance = float(self.pca.explained_variance_ratio_.sum())
        else:
            self.method = f"no reduction needed ({Z.shape[1]} features <= {self.n_qubits} qubits)"
        self.angles.fit(Z)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.standardiser.transform(np.asarray(X, dtype=float))
        if self.pca is not None:
            Z = self.pca.transform(Z)
        A = self.angles.transform(Z)
        # Angles wrap, so a value outside the fitted range would alias onto a
        # different rotation. Clip, and count how often it happened.
        outside = np.sum((A < ANGLE_RANGE[0]) | (A > ANGLE_RANGE[1]))
        self.n_clipped_ += int(outside)
        return np.clip(A, *ANGLE_RANGE)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def describe(self) -> dict:
        return {
            "method": self.method,
            "n_qubits": self.n_qubits,
            "angle_range": [float(ANGLE_RANGE[0]), float(ANGLE_RANGE[1])],
            "explained_variance_ratio": self.explained_variance,
            "values_clipped_to_range": self.n_clipped_,
            "fitted_on": "training rows only",
        }


def load_task(target: str, task: str) -> QuantumTaskData:
    """Load one target with Task 6's own contract, splits and folds.

    The encoder fitted here is the *development-set* one, used for the final
    model. Each CV fold fits its own encoder inside
    ``encode_fold`` so no fold ever sees its validation rows during fitting.
    """
    dataset = load_and_validate()
    frame, features = build_task_frame(dataset, target)

    holdout = chronological_holdout(frame)
    dev = frame.loc[holdout.dev_index].reset_index(drop=True)
    folds = expanding_window_folds(dev)

    X_dev_raw = dev[features].to_numpy(float)
    X_test_raw = frame.loc[holdout.test_index, features].to_numpy(float)
    y_dev = dev[target].to_numpy(float)
    y_test = frame.loc[holdout.test_index, target].to_numpy(float)

    encoder = AngleEncoder()
    X_dev = encoder.fit_transform(X_dev_raw)
    X_test = encoder.transform(X_test_raw)

    data = QuantumTaskData(
        target=target,
        task=task,
        feature_names=features,
        X_dev=X_dev,
        y_dev=y_dev,
        X_test=X_test,
        y_test=y_test,
        dev_index=np.asarray(holdout.dev_index),
        test_index=np.asarray(holdout.test_index),
        folds=folds,
        holdout=holdout.to_metadata(),
        reduction=encoder.describe(),
    )

    if task == "regression":
        # Circuit outputs are bounded in [-1, 1], so a target in the 90-100 s
        # range cannot be produced directly. Standardise on development rows
        # only and invert before any metric is computed.
        data.y_dev_raw, data.y_test_raw = y_dev, y_test
        data.target_mean = float(y_dev.mean())
        data.target_std = float(y_dev.std())
        data.y_dev = (y_dev - data.target_mean) / data.target_std
        data.y_test = (y_test - data.target_mean) / data.target_std
        data.notes.append(
            f"Target standardised on development rows only (mean {data.target_mean:.3f} s, "
            f"sd {data.target_std:.3f} s) and inverted before metrics."
        )

    data.notes.append(
        f"{len(features)} selected features -> {encoder.describe()['method']}"
        + (f", retaining {encoder.explained_variance:.1%} of variance" if encoder.explained_variance else "")
    )
    return data


def encode_fold(data: QuantumTaskData, fold) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-fit the encoder on this fold's training rows and encode both sides.

    The raw feature values are re-read from the contract so the fold's encoder
    never inherits the development-set statistics.
    """
    dataset = load_and_validate()
    frame, features = build_task_frame(dataset, data.target)
    dev = frame.loc[data.dev_index].reset_index(drop=True)

    X_raw = dev[features].to_numpy(float)
    y = dev[data.target].to_numpy(float)
    if data.task == "regression":
        y = (y - data.target_mean) / data.target_std

    encoder = AngleEncoder(data.n_qubits)
    X_train = encoder.fit_transform(X_raw[fold.train_index])
    X_val = encoder.transform(X_raw[fold.val_index])
    return X_train, y[fold.train_index], X_val, y[fold.val_index]


def class_weights(y: np.ndarray) -> tuple[float, float]:
    """Balanced weights for a binary target: ``n / (2 * n_class)``.

    Pit events are about 5% of laps, so an unweighted loss is minimised by
    never predicting a stop.
    """
    y = np.asarray(y)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0
    n = n_pos + n_neg
    return n / (2.0 * n_neg), n / (2.0 * n_pos)
