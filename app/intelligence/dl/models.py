"""
app.intelligence.dl.models
===========================

Deep neural network architectures for the two Task 7 targets.

Design rationale (this matters more than usual here, so it is stated in the
code rather than only in the report):

The Task 5 matrix has **995 rows**. After the chronological holdout and the
expanding-window folds, a single training fold can be a few hundred rows.
That is a *small-data* regime, and it dictates everything below:

* **Depth/width are chosen on validation data, starting small.** The search
  tries 1-3 hidden layers (the specification's 128-64-32 included) and keeps
  what validates best; the report states the parameters-to-rows ratio.
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
import numpy as np  # noqa: E402


def _optimizer(name: str, learning_rate: float):
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    return keras.optimizers.Adam(learning_rate=learning_rate)


def _hidden_stack(n_features: int, hidden_units, dropout: float, l2: float) -> list:
    reg = keras.regularizers.l2(l2) if l2 else None
    layers: list = [keras.layers.Input(shape=(n_features,), name="features")]
    for i, units in enumerate(hidden_units):
        layers.append(keras.layers.Dense(units, activation="relu", kernel_regularizer=reg,
                                         name=f"hidden_{i + 1}"))
        layers.append(keras.layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    return layers


def weighted_binary_crossentropy(neg_weight: float, pos_weight: float):
    """Binary cross-entropy with the class weights applied **inside** the loss.

    Why not ``model.fit(class_weight=...)``: measured on this installation
    (Keras 3.15.1, PyTorch backend), neither ``class_weight`` nor
    ``sample_weight`` weights the loss as documented. For a constant prediction
    of 0.05 on a 5%-positive set with weights {0: 0.5, 1: 10}, Keras reported a
    weighted loss of 0.1419 where mean(w * loss) is 1.1730 - lower even than the
    unweighted loss (0.1642) - and a class-weighted fit left the predictions
    essentially where an unweighted fit put them. Earlier Task 7 runs passed
    ``class_weight`` to ``fit`` and were therefore effectively unweighted.
    Weighting inside the loss is unambiguous, and ``tests/test_dl_training.py``
    asserts it actually moves the predictions.
    """
    def class_weighted_binary_crossentropy(y_true, y_pred):
        y_true = keras.ops.reshape(y_true, keras.ops.shape(y_pred))
        y_pred = keras.ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        w = keras.ops.where(y_true > 0.5, pos_weight, neg_weight)
        per = -(y_true * keras.ops.log(y_pred) + (1.0 - y_true) * keras.ops.log(1.0 - y_pred))
        return keras.ops.mean(w * per, axis=-1)
    return class_weighted_binary_crossentropy


def build_regression_mlp(
    n_features: int,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    loss: str = "mse",
    class_weight: dict | None = None,  # accepted for a uniform builder signature; unused
) -> keras.Model:
    """MLP with a linear output head for ``target_laptime`` (seconds).

    ``loss`` is ``"mse"`` or ``"huber"``. Huber is quadratic for small errors and
    linear for large ones, so a handful of anomalous laps (traffic, a lock-up)
    pulls the fit less than under MSE.
    """
    layers = _hidden_stack(n_features, hidden_units, dropout, l2)
    layers.append(keras.layers.Dense(1, activation="linear", name="laptime_seconds"))
    model = keras.Sequential(layers, name="laptime_mlp")
    model.compile(
        optimizer=_optimizer(optimizer, learning_rate),
        loss=keras.losses.Huber(name="huber") if loss == "huber" else "mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_classification_mlp(
    n_features: int,
    hidden_units: tuple[int, ...] = (32, 16),
    dropout: float = 0.3,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    loss: str = "binary_crossentropy",  # accepted for a uniform builder signature
    class_weight: dict | None = None,
) -> keras.Model:
    """MLP with a sigmoid output head for ``target_pit_next_lap``.

    With ``class_weight`` given, the rare pit class is up-weighted inside the
    loss (see ``weighted_binary_crossentropy``); without it, plain binary
    cross-entropy. Whether to weight at all is decided on validation data by the
    hyperparameter search - it is not assumed. No resampling is ever done, so no
    synthetic rows enter this time-ordered panel.
    """
    layers = _hidden_stack(n_features, hidden_units, dropout, l2)
    layers.append(keras.layers.Dense(1, activation="sigmoid", name="pit_probability"))
    model = keras.Sequential(layers, name="pit_decision_mlp")
    model.compile(
        optimizer=_optimizer(optimizer, learning_rate),
        loss=(weighted_binary_crossentropy(class_weight[0], class_weight[1])
              if class_weight else "binary_crossentropy"),
        metrics=["accuracy"],
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
        "trainable_parameters": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
        "non_trainable_parameters": int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)),
        "optimizer": type(model.optimizer).__name__,
        "loss": (model.loss if isinstance(model.loss, str)
                 else getattr(model.loss, "__name__", None) or getattr(model.loss, "name", None)
                 or type(model.loss).__name__),
    }
