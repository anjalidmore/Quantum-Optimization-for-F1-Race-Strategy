"""
app.intelligence.qml.kernel_svm
===============================

A Quantum Kernel SVM for the pit decision — the second quantum approach, and a
structurally different one from the VQC.

Nothing is trained on the circuit here. The circuit only measures how similar
two laps are once they have been embedded into quantum states:

    K(a, b) = |<phi(b) | phi(a)>|^2

computed as the probability of measuring all zeros after applying the feature
map for ``a`` and then its adjoint for ``b``. That matrix is handed to
scikit-learn's SVC with ``kernel="precomputed"``, so the SVM is the classical
part and the kernel is the quantum part.

Two practical notes:

* The feature map is ``AngleEmbedding`` plus a ring of CZ gates, repeated
  ``reps`` times. The entanglers are what make this kernel different from a
  classical cosine-type kernel; without them the fidelity factorises per qubit.
* Cost is O(n²) circuit evaluations. On 4 qubits with PennyLane's broadcasting
  the whole 635×635 development matrix takes about a second, so no subsampling
  was needed — the numbers below are on the full split.
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from sklearn.svm import SVC


def _feature_map(x, n_qubits: int, reps: int) -> None:
    for _ in range(reps):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.CZ(wires=[i, (i + 1) % n_qubits])


def make_kernel(n_qubits: int, reps: int = 2):
    """Return ``kernel(A, B) -> matrix`` using one broadcast circuit call."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def fidelity(x1, x2):
        _feature_map(x1, n_qubits, reps)
        qml.adjoint(_feature_map)(x2, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        left = np.repeat(A, len(B), axis=0)          # every (a, b) pair, in one call
        right = np.tile(B, (len(A), 1))
        probs = np.asarray(fidelity(left, right), dtype=float)
        return probs[:, 0].reshape(len(A), len(B))   # P(all zeros) == |<phi(b)|phi(a)>|^2

    return kernel


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_qubits: int,
    reps: int = 2,
    C: float = 1.0,
    seed: int = 42,
) -> dict:
    """Fit an SVC on the quantum kernel. ``class_weight='balanced'`` for the imbalance."""
    kernel = make_kernel(n_qubits, reps)
    t0 = time.perf_counter()
    K = kernel(X, X)
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced", probability=False, random_state=seed)
    svc.fit(K, y)
    train_seconds = time.perf_counter() - t0

    return {
        "svc": svc,
        "X_train": np.asarray(X, dtype=float),      # needed to rebuild the kernel at predict time
        "n_qubits": n_qubits,
        "reps": reps,
        "C": C,
        "train_seconds": train_seconds,
        # The SVM's learned parameters are its dual coefficients plus the
        # intercept; the circuit itself has none to train.
        "n_parameters": int(svc.dual_coef_.size + 1),
        "n_support_vectors": int(svc.support_vectors_.shape[0]) if hasattr(svc, "support_vectors_") else int(sum(svc.n_support_)),
        "kernel_matrix_shape": list(K.shape),
        "seed": seed,
    }


def decision_scores(model: dict, X: np.ndarray) -> np.ndarray:
    """SVM decision values. Higher means more pit-like.

    These are margins, not probabilities. They are used for ranking metrics
    (ROC-AUC, PR-AUC) directly, and squashed only where a 0-1 score is needed.
    """
    kernel = make_kernel(model["n_qubits"], model["reps"])
    K = kernel(np.asarray(X, dtype=float), model["X_train"])
    return np.asarray(model["svc"].decision_function(K), dtype=float)


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """A 0-1 score from the margin, so the same threshold tuning applies.

    This is a monotone squash of the decision value, **not** a calibrated
    probability: it preserves the ranking the margin gives and nothing more.
    """
    return 1.0 / (1.0 + np.exp(-decision_scores(model, X)))
