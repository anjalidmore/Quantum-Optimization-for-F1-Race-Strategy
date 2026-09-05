"""
app.intelligence.dl.models
===========================

Deep neural network architectures for the two Task 7 targets.

Design rationale (this matters more than usual here, so it is stated in the
code rather than only in the report):

The Task 5 matrix has **995 rows**. After the chronological holdout and the
expanding-window folds, a single training fold can be a few hundred rows.
That is a *small-data* regime, and it dictates everything below:

* **Depth/width are deliberately small.** Two hidden layers, 16-64 units.
  A network with more parameters than it has training rows will memorise the
  training laps and generalise worse than a decision tree - the honest
  expectation here is not that "deeper is better".
* **Dropout on every hidden layer.** With this little data, dropout is the
  cheapest effective regulariser.
* **L2 weight decay** in addition to dropout, because the regression target
  has 45 input features (28 of them driver/team one-hots) and unregularised
  weights on sparse indicators overfit almost immediately.
* **The classifier is smaller still.** Its target has 8 features and only
  48 positive examples in the whole dataset; anything larger is fitting noise.

Output heads follow the task: linear for lap-time regression, sigmoid for
binary pit-decision.
"""
from __future__ import annotations

# Keras 3 is backend-agnostic. TensorFlow publishes no wheel for the Python
# version this project targets, so the torch backend is selected here, before
# keras is imported. The Keras API used below is identical either way.
from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()

import keras  # noqa: E402


def build_regression_mlp(
    n_features: int,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """MLP with a linear output head for ``target_laptime`` (seconds)."""
    layers: list = [keras.layers.Input(shape=(n_features,), name="features")]
    for i, units in enumerate(hidden_units):
        layers.append(
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2),
                name=f"hidden_{i + 1}",
            )
        )
        layers.append(keras.layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    layers.append(keras.layers.Dense(1, activation="linear", name="laptime_seconds"))

    model = keras.Sequential(layers, name="laptime_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_classification_mlp(
    n_features: int,
    hidden_units: tuple[int, ...] = (32, 16),
    dropout: float = 0.3,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """MLP with a sigmoid output head for ``target_pit_next_lap``.

    Trained with ``binary_crossentropy``. Class imbalance (4.8% positives) is
    handled at ``fit`` time via ``class_weight`` rather than by resampling, so
    no synthetic rows ever enter this time-ordered panel.
    """
    layers: list = [keras.layers.Input(shape=(n_features,), name="features")]
    for i, units in enumerate(hidden_units):
        layers.append(
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2),
                name=f"hidden_{i + 1}",
            )
        )
        layers.append(keras.layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    layers.append(keras.layers.Dense(1, activation="sigmoid", name="pit_probability"))

    model = keras.Sequential(layers, name="pit_decision_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


BUILDERS = {
    "target_laptime": build_regression_mlp,
    "target_pit_next_lap": build_classification_mlp,
}


def architecture_summary(model: keras.Model) -> dict:
    """A JSON-serialisable description of the built network, for the
    hyperparameter report and the model registry."""
    layers = []
    for layer in model.layers:
        entry = {"name": layer.name, "type": type(layer).__name__}
        if isinstance(layer, keras.layers.Dense):
            entry["units"] = int(layer.units)
            entry["activation"] = layer.activation.__name__
        elif isinstance(layer, keras.layers.Dropout):
            entry["rate"] = float(layer.rate)
        layers.append(entry)
    return {
        "name": model.name,
        "layers": layers,
        "total_parameters": int(model.count_params()),
        "optimizer": type(model.optimizer).__name__,
        "loss": model.loss if isinstance(model.loss, str) else type(model.loss).__name__,
    }
