"""
app.intelligence.qml.vqc
========================

A Variational Quantum Classifier for ``target_pit_next_lap``.

The circuit, on ``default.qubit`` (a noiseless simulator — there is no quantum
hardware anywhere in this project):

    AngleEmbedding(x)  ->  StronglyEntanglingLayers(weights)  ->  <PauliZ(0)>

``<PauliZ(0)>`` lives in [-1, 1], so a trainable scale and bias turn it into a
probability through a sigmoid. Training minimises a **class-weighted** binary
cross-entropy: pit events are about 5% of laps, and an unweighted loss is
minimised by a model that never predicts a stop.

Trainable parameters: ``3 * layers * qubits`` circuit angles, plus the scale and
bias. With 4 qubits and 2 layers that is 26 numbers — which is why the classical
baseline in ``baselines.py`` is sized to match rather than being a full model.
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

EPS = 1e-7


def make_circuit(n_qubits: int, n_layers: int):
    """The QNode plus the weight shape it expects."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return circuit, shape


def probability(circuit, x, weights, scale, bias):
    """Sigmoid of the scaled expectation value: a pit probability in (0, 1)."""
    z = circuit(x, weights)
    return 1.0 / (1.0 + pnp.exp(-(scale * z + bias)))


def weighted_bce(p, y, w_neg: float, w_pos: float):
    p = pnp.clip(p, EPS, 1.0 - EPS)
    return -pnp.mean(w_pos * y * pnp.log(p) + w_neg * (1.0 - y) * pnp.log(1.0 - p))


def n_parameters(n_qubits: int, n_layers: int) -> int:
    return int(3 * n_layers * n_qubits) + 2


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    learning_rate: float,
    epochs: int = 60,
    class_weight: tuple[float, float] = (1.0, 1.0),
    seed: int = 42,
    log=None,
) -> dict:
    """Full-batch Adam training. Deterministic for a given seed.

    Full batch rather than mini-batch on purpose: broadcasting the whole
    training set through the simulator in one call costs about as much as one
    sample, so batching would only add noise and code.
    """
    circuit, shape = make_circuit(n_qubits, n_layers)
    rng = np.random.default_rng(seed)

    weights = pnp.array(rng.normal(0.0, 0.1, shape), requires_grad=True)
    scale = pnp.array(1.0, requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)

    Xa = pnp.array(X, requires_grad=False)
    ya = pnp.array(y, requires_grad=False)
    w_neg, w_pos = class_weight

    def cost(weights, scale, bias):
        return weighted_bce(probability(circuit, Xa, weights, scale, bias), ya, w_neg, w_pos)

    opt = qml.AdamOptimizer(stepsize=learning_rate)
    history: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        (weights, scale, bias), loss = opt.step_and_cost(cost, weights, scale, bias)
        history.append(float(loss))
        if log and (epoch % 10 == 0 or epoch == epochs - 1):
            log.info("    epoch %3d/%d  weighted BCE %.4f", epoch + 1, epochs, float(loss))
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
        "class_weight": [float(w_neg), float(w_pos)],
        "n_parameters": n_parameters(n_qubits, n_layers),
        "seed": seed,
    }


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """Pit probability per row, from a trained (or reloaded) model dict."""
    circuit, _ = make_circuit(model["n_qubits"], model["n_layers"])
    weights = pnp.array(model["weights"], requires_grad=False)
    z = np.asarray(circuit(pnp.array(X, requires_grad=False), weights), dtype=float)
    return 1.0 / (1.0 + np.exp(-(model["scale"] * z + model["bias"])))
