"""
app.intelligence.qml.vqr
========================

A Variational Quantum Regressor for ``target_laptime``.

Same circuit family as the classifier, different head:

    AngleEmbedding(x)  ->  StronglyEntanglingLayers(weights)  ->  <PauliZ(0)>
    prediction = scale * <PauliZ(0)> + bias      (standardised lap time)

``<PauliZ(0)>`` is bounded in [-1, 1], so the network cannot emit "97.5 seconds"
directly. The target is standardised on development rows only (see
``data.load_task``) and every prediction is inverted back to seconds before any
metric is computed — so the MAE and R² below are directly comparable to Task 6's
and Task 7's.

Loss is plain mean squared error on the standardised target.
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def make_circuit(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return circuit, shape


def n_parameters(n_qubits: int, n_layers: int) -> int:
    return int(3 * n_layers * n_qubits) + 2


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    learning_rate: float,
    epochs: int = 80,
    seed: int = 42,
    log=None,
) -> dict:
    """Full-batch Adam on the standardised target. Deterministic for a seed."""
    circuit, shape = make_circuit(n_qubits, n_layers)
    rng = np.random.default_rng(seed)

    weights = pnp.array(rng.normal(0.0, 0.1, shape), requires_grad=True)
    # Start the head at the data's own scale so the first epochs are not spent
    # discovering that lap times vary by about one standard deviation.
    scale = pnp.array(1.0, requires_grad=True)
    bias = pnp.array(float(np.mean(y)), requires_grad=True)

    Xa = pnp.array(X, requires_grad=False)
    ya = pnp.array(y, requires_grad=False)

    def cost(weights, scale, bias):
        pred = scale * circuit(Xa, weights) + bias
        return pnp.mean((pred - ya) ** 2)

    opt = qml.AdamOptimizer(stepsize=learning_rate)
    history: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        (weights, scale, bias), loss = opt.step_and_cost(cost, weights, scale, bias)
        history.append(float(loss))
        if log and (epoch % 20 == 0 or epoch == epochs - 1):
            log.info("    epoch %3d/%d  MSE (standardised) %.4f", epoch + 1, epochs, float(loss))
    train_seconds = time.perf_counter() - t0

    return {
        "weights": np.array(weights),
        "scale": float(scale),
        "bias": float(bias),
        "loss_history": history,
        "train_seconds": train_seconds,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "n_parameters": n_parameters(n_qubits, n_layers),
        "seed": seed,
    }


def predict_standardised(model: dict, X: np.ndarray) -> np.ndarray:
    """Predictions in standardised units; the caller inverts them to seconds."""
    circuit, _ = make_circuit(model["n_qubits"], model["n_layers"])
    weights = pnp.array(model["weights"], requires_grad=False)
    z = np.asarray(circuit(pnp.array(X, requires_grad=False), weights), dtype=float)
    return model["scale"] * z + model["bias"]
