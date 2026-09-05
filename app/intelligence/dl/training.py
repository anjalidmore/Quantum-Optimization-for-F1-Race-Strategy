"""
app.intelligence.dl.training
============================

Fitting a Keras model on one fold, with the leakage-safe preprocessing the
Task 5 contract demands.

The contract states scaling must be fit *inside* each fold
(``preprocessing_contract.scaling``: "NOT applied here. Fit the scaler inside
each CV fold to avoid leakage."). ``fit_fold`` therefore constructs a fresh
``StandardScaler`` from the training rows only and applies it to the
validation rows - the validation statistics never influence the transform.

Overfitting prevention is applied in three places and each is reported:
dropout (in ``models``), L2 weight decay (in ``models``), and early stopping
on validation loss with ``restore_best_weights=True`` (here).

**Target standardisation (regression only).** ``target_laptime`` has mean
~91 s but a standard deviation of ~0.56 s. A linear output head initialised
near zero, trained under MSE with L2 weight decay, cannot climb to 91 within
any sane epoch budget - it converges to a large constant error while the
signal of interest (sub-second variation) is invisible to the optimiser.
The target is therefore standardised using the *training rows only* and
predictions are inverse-transformed before any metric is computed, so every
reported number is in real seconds. Tree models are scale-invariant and need
none of this, which is precisely why the network needs it to be compared
fairly rather than handicapped by an implementation detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RANDOM_STATE = 42


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy and the active Keras backend together."""
    keras.utils.set_random_seed(seed)


@dataclass
class FoldFit:
    model: keras.Model
    scaler: StandardScaler
    history: dict = field(default_factory=dict)
    epochs_run: int = 0
    best_epoch: int = 0
    y_scaler: StandardScaler | None = None


def scale_fit_transform(
    X_train: np.ndarray, X_other: np.ndarray, numeric_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a scaler on the training rows only and apply it to both sets.

    Binary indicator columns (per the Task 5 contract's
    ``binary_features_no_scaling_needed``) are passed through untouched -
    standardising a 0/1 dummy destroys its interpretability without helping
    the optimiser.
    """
    scaler = StandardScaler()
    Xt = X_train.astype("float32").copy()
    Xo = X_other.astype("float32").copy()
    if numeric_mask.any():
        Xt[:, numeric_mask] = scaler.fit_transform(X_train[:, numeric_mask])
        Xo[:, numeric_mask] = scaler.transform(X_other[:, numeric_mask])
    return Xt, Xo, scaler


def fit_fold(
    build_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    numeric_mask: np.ndarray,
    *,
    hidden_units: tuple[int, ...],
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int = 200,
    patience: int = 20,
    class_weight: dict | None = None,
    scale_target: bool = False,
    seed: int = RANDOM_STATE,
    verbose: int = 0,
) -> FoldFit:
    set_seeds(seed)
    Xt, Xv, scaler = scale_fit_transform(X_train, X_val, numeric_mask)

    y_scaler = None
    yt, yv = y_train.astype("float32"), y_val.astype("float32")
    if scale_target:
        y_scaler = StandardScaler()
        yt = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype("float32")
        yv = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype("float32")

    model = build_fn(
        n_features=Xt.shape[1],
        hidden_units=hidden_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
    )
    hist = model.fit(
        Xt,
        yt,
        validation_data=(Xv, yv),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early],
        class_weight=class_weight,
        shuffle=True,
        verbose=verbose,
    )

    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    val_loss = history.get("val_loss", [])
    best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else 0

    return FoldFit(
        model=model,
        scaler=scaler,
        history=history,
        epochs_run=len(val_loss),
        best_epoch=best_epoch,
        y_scaler=y_scaler,
    )


def balanced_class_weight(y: np.ndarray) -> dict[int, float]:
    """sklearn's 'balanced' weighting, computed explicitly so the value that
    went into training is recorded in the report rather than hidden inside a
    library default."""
    y = np.asarray(y).astype(int)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): float(n / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}


def predict(
    model: keras.Model,
    scaler: StandardScaler,
    X: np.ndarray,
    numeric_mask: np.ndarray,
    y_scaler: StandardScaler | None = None,
) -> np.ndarray:
    """Apply the fold's fitted feature scaler, predict, and (for regression)
    invert the target standardisation so the result is in real seconds."""
    Xs = X.astype("float32").copy()
    if numeric_mask.any():
        Xs[:, numeric_mask] = scaler.transform(X[:, numeric_mask])
    out = np.asarray(model.predict(Xs, verbose=0)).ravel()
    if y_scaler is not None:
        out = y_scaler.inverse_transform(out.reshape(-1, 1)).ravel()
    return out
