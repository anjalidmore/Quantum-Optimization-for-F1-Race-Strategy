"""
Quantum ML tests.

The circuits themselves are the easy part to trust — PennyLane simulates them
exactly. What needs guarding is everything around them: that the quantum models
are scored on the same laps as the classical ones, that nothing about the test
laps leaks in through a scaler or a PCA, that saved weights really reproduce the
predictions they were saved with, and that a seeded run is reproducible.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.core.paths import QML_METRICS_JSON, QML_MODELS_DIR
from app.intelligence.qml import data as qdata
from app.intelligence.qml import kernel_svm, vqc, vqr

_built = pytest.mark.skipif(not QML_METRICS_JSON.exists(), reason="QML not built; run scripts/run_qml.py")

N_Q = 4


# ---------------------------------------------------------------------------
# Encoding: fitted on training rows only, and bounded for angle embedding
# ---------------------------------------------------------------------------
def test_encoder_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (60, 6))
    X_test = rng.normal(50, 1, (20, 6))          # a wildly different distribution

    enc = qdata.AngleEncoder(N_Q).fit(X_train)
    mean_before = enc.standardiser.mean_.copy()
    components_before = enc.pca.components_.copy()

    enc.transform(X_test)
    assert np.array_equal(enc.standardiser.mean_, mean_before), "transform must not refit the standardiser"
    assert np.array_equal(enc.pca.components_, components_before), "transform must not refit the PCA"


def test_encoded_features_stay_inside_the_angle_range():
    rng = np.random.default_rng(1)
    enc = qdata.AngleEncoder(N_Q).fit(rng.normal(0, 1, (80, 6)))
    A = enc.transform(rng.normal(20, 5, (30, 6)))   # far outside the fitted range
    assert A.shape == (30, N_Q)
    assert A.min() >= 0.0 and A.max() <= np.pi, "angles must be clipped to [0, pi]"
    assert enc.n_clipped_ > 0, "out-of-range values should be counted, not silently passed"


def test_encoder_reports_how_it_reduced_the_features():
    rng = np.random.default_rng(2)
    reduced = qdata.AngleEncoder(N_Q).fit(rng.normal(0, 1, (50, 12))).describe()
    assert "PCA" in reduced["method"] and reduced["explained_variance_ratio"] is not None
    assert qdata.AngleEncoder(N_Q).fit(rng.normal(0, 1, (50, 3))).describe()["explained_variance_ratio"] is None


def test_balanced_class_weights_favour_the_rare_class():
    y = np.r_[np.zeros(95), np.ones(5)]
    w_neg, w_pos = qdata.class_weights(y)
    assert w_pos > w_neg
    assert np.isclose(w_neg * 95 + w_pos * 5, 100.0)      # weights rebalance the two halves


# ---------------------------------------------------------------------------
# Circuits: output shape, and the same seed giving the same answer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_rows", [1, 7])
def test_vqc_returns_one_probability_per_row(n_rows):
    X = np.random.default_rng(3).uniform(0, np.pi, (n_rows, N_Q))
    model = vqc.train(X, np.array([0, 1] * n_rows)[:n_rows].astype(float),
                      n_qubits=N_Q, n_layers=1, learning_rate=0.1, epochs=2)
    p = vqc.predict_proba(model, X)
    assert p.shape == (n_rows,)
    assert np.all((p > 0) & (p < 1)), "a sigmoid output must be a probability"


def test_vqr_returns_one_prediction_per_row():
    X = np.random.default_rng(4).uniform(0, np.pi, (9, N_Q))
    model = vqr.train(X, np.linspace(-1, 1, 9), n_qubits=N_Q, n_layers=1, learning_rate=0.1, epochs=2)
    assert vqr.predict_standardised(model, X).shape == (9,)


def test_parameter_count_matches_the_circuit_shape():
    # 3 angles per qubit per layer, plus the trainable scale and bias
    assert vqc.n_parameters(4, 2) == 3 * 2 * 4 + 2
    model = vqc.train(np.random.default_rng(5).uniform(0, np.pi, (6, N_Q)),
                      np.array([0.0, 1, 0, 1, 0, 1]), n_qubits=N_Q, n_layers=2,
                      learning_rate=0.1, epochs=1)
    assert model["weights"].size + 2 == vqc.n_parameters(N_Q, 2)


def test_the_seed_makes_training_deterministic():
    X = np.random.default_rng(6).uniform(0, np.pi, (40, N_Q))
    y = (np.random.default_rng(7).random(40) < 0.3).astype(float)
    kwargs = dict(n_qubits=N_Q, n_layers=2, learning_rate=0.1, epochs=5)
    a = vqc.train(X, y, seed=42, **kwargs)
    b = vqc.train(X, y, seed=42, **kwargs)
    c = vqc.train(X, y, seed=7, **kwargs)
    assert np.allclose(a["weights"], b["weights"]), "same seed must give the same weights"
    assert np.allclose(vqc.predict_proba(a, X), vqc.predict_proba(b, X))
    assert not np.allclose(a["weights"], c["weights"]), "a different seed should explore differently"


def test_quantum_kernel_is_a_valid_similarity_matrix():
    X = np.random.default_rng(8).uniform(0, np.pi, (12, N_Q))
    K = kernel_svm.make_kernel(N_Q, reps=2)(X, X)
    assert K.shape == (12, 12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6), "a state has fidelity 1 with itself"
    assert np.allclose(K, K.T, atol=1e-8), "fidelity is symmetric"
    assert K.min() >= -1e-9 and K.max() <= 1 + 1e-9


# ---------------------------------------------------------------------------
# Saved artifacts: reload and reproduce
# ---------------------------------------------------------------------------
@_built
@pytest.mark.parametrize("name,module", [("vqc", vqc), ("vqr", vqr)])
def test_saved_weights_reload_and_reproduce_their_predictions(name, module):
    config = json.loads((QML_MODELS_DIR / f"{name}_config.json").read_text())
    weights = np.load(QML_MODELS_DIR / config["weights_file"])
    model = {"weights": weights, "scale": config["scale"], "bias": config["bias"],
             "n_qubits": config["n_qubits"], "n_layers": config["n_layers"]}

    X = np.random.default_rng(9).uniform(0, np.pi, (5, config["n_qubits"]))
    predict = module.predict_proba if name == "vqc" else module.predict_standardised
    first = predict(model, X)
    second = predict({**model, "weights": np.load(QML_MODELS_DIR / config["weights_file"])}, X)
    assert np.allclose(first, second, atol=1e-12), "a reloaded model must predict identically"
    assert weights.size + 2 == config["n_parameters"]


@_built
def test_committed_run_used_the_same_split_as_task6():
    from app.intelligence.ml.data_contract import build_task_frame, load_and_validate
    from app.intelligence.ml.splits import chronological_holdout

    results = json.loads(QML_METRICS_JSON.read_text())
    frame, _ = build_task_frame(load_and_validate(), "target_laptime")
    task6 = chronological_holdout(frame).to_metadata()
    for target, r in results["targets"].items():
        assert r["holdout"]["test_rows"] == task6["test_rows"], target
        assert r["holdout"]["test_laps"] == task6["test_laps"], target


@_built
def test_no_leakage_and_no_fabricated_numbers_in_the_committed_run():
    results = json.loads(QML_METRICS_JSON.read_text())
    assert "default.qubit" in results["simulator"]
    for target, r in results["targets"].items():
        assert r["encoding"]["fitted_on"] == "training rows only"
        for model in r["quantum"] + r["classical_matched"]:
            assert model["test_metrics"]["n"] == r["n_test"]
            assert model["n_parameters"] > 0
            # every reported CV figure must come with the folds that produced it
            summary = model.get("cv_summary") or {}
            assert summary.get("n_folds_total") == r["n_folds"]
            for key, metric in summary.items():
                if isinstance(metric, dict) and metric.get("mean") is not None:
                    assert metric["n_folds"] == r["n_folds"], f"{target}/{model['model']}/{key}"


@_built
def test_every_promised_artifact_exists():
    from app.core.paths import ARTIFACTS_DIR, ML_REPORTS_DIR

    results = json.loads(QML_METRICS_JSON.read_text())
    for rel in results["figures"].values():
        assert (ARTIFACTS_DIR / rel).exists(), rel
    assert (ML_REPORTS_DIR / "classical_vs_quantum_report.md").exists()
    for name in ("vqc_weights.npy", "vqr_weights.npy", "vqc_config.json", "vqr_config.json",
                 "quantum_kernel_svc.joblib"):
        assert (QML_MODELS_DIR / name).exists(), name
