"""
Task 7 — deep learning tests.

Mirrors the guarantees ``test_ml_*.py`` asserts for Task 6: the data contract
holds, no leakage column reaches the model, the split never trains on the
future, a saved model reloads and predicts, and the committed artifacts match
each other.
"""
from __future__ import annotations

import json

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()

import numpy as np
import pytest

from app.core.paths import (
    DL_COMPARISON_JSON,
    DL_HISTORY_JSON,
    DL_METRICS_JSON,
    DL_MODELS_DIR,
    ML_MODEL_REGISTRY_JSON,
)
from app.intelligence.dl import models, persistence, training, tuning
from app.intelligence.dl import pipeline as dl_pipeline
from app.intelligence.ml.data_contract import build_task_frame, load_and_validate
from app.intelligence.ml.splits import chronological_holdout, expanding_window_folds

TARGETS = ("target_laptime", "target_pit_next_lap")

_artifacts = pytest.mark.skipif(
    not DL_METRICS_JSON.exists(), reason="Task 7 not built; run scripts/build_all.py"
)


# --------------------------------------------------------------------------
# Contract — Task 7 must inherit Task 6's guarantees, not weaken them
# --------------------------------------------------------------------------
def test_dl_uses_the_same_feature_contract_as_task6():
    dataset = load_and_validate()
    for target in TARGETS:
        frame, features = build_task_frame(dataset, target)
        assert features == dataset.contract.selected_features(target)


def test_no_leakage_column_enters_the_dl_feature_set():
    """The same check Task 6 performs — a deep model must not be given the
    columns Task 5 excluded as leakage."""
    dataset = load_and_validate()
    leakage = set(dataset.contract.leakage_columns)
    for target in TARGETS:
        _, features = build_task_frame(dataset, target)
        assert not (set(features) & leakage), f"{target} received leakage columns"


def test_no_target_is_its_own_predictor():
    dataset = load_and_validate()
    targets = set(dataset.contract.target_names)
    for target in TARGETS:
        _, features = build_task_frame(dataset, target)
        assert not (set(features) & targets)


def test_dl_split_never_trains_on_the_future():
    dataset = load_and_validate()
    frame, _ = build_task_frame(dataset, "target_laptime")
    holdout = chronological_holdout(frame, test_fraction=0.2)
    assert max(holdout.dev_laps) < min(holdout.test_laps)
    for fold in expanding_window_folds(frame.loc[holdout.dev_index], n_folds=4):
        assert max(fold.train_laps) < min(fold.val_laps)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
def test_regression_head_is_linear_classification_head_is_sigmoid():
    assert models.build_regression_mlp(8).layers[-1].activation.__name__ == "linear"
    assert models.build_classification_mlp(8).layers[-1].activation.__name__ == "sigmoid"


def test_architecture_summary_is_json_serialisable():
    s = models.architecture_summary(models.build_regression_mlp(8))
    json.dumps(s)
    assert s["total_parameters"] > 0


def test_search_space_is_fully_enumerated_not_sampled():
    for space in (tuning.REGRESSION_SPACE, tuning.CLASSIFICATION_SPACE):
        expected = (len(space.hidden_units) * len(space.dropout)
                    * len(space.learning_rate) * len(space.batch_size))
        assert len(space.combinations()) == expected


# --------------------------------------------------------------------------
# Training mechanics
# --------------------------------------------------------------------------
def test_scaler_is_fit_on_training_rows_only():
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(0, 1, (50, 3)), rng.normal(50, 1, (20, 3))
    mask = np.ones(3, dtype=bool)
    Xt, Xv, _ = training.scale_fit_transform(Xtr, Xva, mask)
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-5)
    assert abs(Xv.mean()) > 10, "validation rows influenced the transform"


def test_binary_indicator_columns_are_not_scaled():
    X = np.column_stack([np.random.normal(0, 1, 40), np.random.randint(0, 2, 40)])
    Xt, _, _ = training.scale_fit_transform(X, X, np.array([True, False]))
    assert set(np.unique(Xt[:, 1])) <= {0.0, 1.0}


def test_target_scaling_returns_predictions_in_real_units():
    """Guards the defect that produced a 29-second MAE: a lap-time target of
    mean ~91 fed to a linear head initialised near zero."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4)).astype("float32")
    y = (91.0 + rng.normal(0, 0.5, 120)).astype("float32")
    mask = np.ones(4, dtype=bool)
    fit = training.fit_fold(
        models.build_regression_mlp, X[:90], y[:90], X[90:], y[90:], mask,
        hidden_units=(8,), dropout=0.0, learning_rate=1e-2, batch_size=16,
        max_epochs=40, patience=10, scale_target=True)
    pred = training.predict(fit.model, fit.scaler, X[90:], mask, fit.y_scaler)
    assert 85 < float(np.mean(pred)) < 97


def test_training_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4)).astype("float32")
    y = rng.normal(size=80).astype("float32")
    mask = np.ones(4, dtype=bool)
    kw = dict(hidden_units=(8,), dropout=0.1, learning_rate=1e-2, batch_size=16,
              max_epochs=8, patience=4)
    a = training.fit_fold(models.build_regression_mlp, X[:60], y[:60], X[60:], y[60:], mask, **kw)
    b = training.fit_fold(models.build_regression_mlp, X[:60], y[:60], X[60:], y[60:], mask, **kw)
    assert np.allclose(
        training.predict(a.model, a.scaler, X[60:], mask),
        training.predict(b.model, b.scaler, X[60:], mask),
    )


def test_balanced_class_weight_upweights_the_minority():
    w = training.balanced_class_weight(np.array([0] * 95 + [1] * 5))
    assert w[1] > w[0]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_saved_model_reloads_and_predicts(tmp_path):
    from sklearn.preprocessing import StandardScaler

    m = models.build_regression_mlp(4)
    sc = StandardScaler().fit(np.random.normal(size=(20, 4)))
    saved = persistence.save(m, sc, list("abcd"), np.ones(4, dtype=bool), "t", tmp_path)
    assert saved.model_path.suffix == persistence.MODEL_EXTENSION
    m2, sc2, ys2, spec = persistence.load("t", tmp_path)
    assert spec["features"] == list("abcd")
    assert m2.predict(np.random.normal(size=(2, 4)).astype("float32"), verbose=0).shape == (2, 1)


@_artifacts
def test_committed_models_reload_and_predict():
    dataset = load_and_validate()
    for target in TARGETS:
        model, scaler, y_scaler, spec = persistence.load(target, DL_MODELS_DIR)
        _, features = build_task_frame(dataset, target)
        assert spec["features"] == features, f"{target}: saved feature order drifted from the contract"
        X = np.zeros((3, len(features)), dtype="float32")
        out = training.predict(model, scaler, X, np.array(spec["numeric_mask"]), y_scaler)
        assert out.shape == (3,) and np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# Committed artifacts — no fabricated results
# --------------------------------------------------------------------------
@_artifacts
def test_metrics_history_and_comparison_agree_with_each_other():
    metrics = json.loads(DL_METRICS_JSON.read_text())
    history = json.loads(DL_HISTORY_JSON.read_text())
    comparison = json.loads(DL_COMPARISON_JSON.read_text())

    for target in TARGETS:
        m = metrics["models"][target]
        h = history[target]
        c = comparison["targets"][target]

        assert h["epochs_run"] == len(h["history"]["loss"]) > 0
        assert h["best_epoch"] <= h["epochs_run"]
        assert m["hyperparameters"] == h["hyperparameters"]

        deep = next(r for r in c["comparison"] if r["family"] == "deep")
        assert deep["metrics"] == m["test_metrics"], (
            f"{target}: the comparison table's deep row disagrees with dl_metrics.json"
        )


@_artifacts
def test_registry_contains_deep_entries_pointing_at_real_files():
    registry = json.loads(ML_MODEL_REGISTRY_JSON.read_text())
    deep = [m for m in registry["models"] if m.get("family") == "deep"]
    assert len(deep) == len(TARGETS), "Task 7 did not register one model per target"
    for entry in deep:
        from app.core.paths import ARTIFACTS_DIR

        assert (ARTIFACTS_DIR / entry["artifact"]).exists(), entry["artifact"]


@_artifacts
def test_registry_has_one_shared_file_not_a_parallel_one():
    """Task 7 must extend the existing registry, not create a second one."""
    registry = json.loads(ML_MODEL_REGISTRY_JSON.read_text())
    families = {m.get("family", "classical") for m in registry["models"]}
    assert "deep" in families and len(families) > 1, (
        "the shared registry should hold both classical and deep entries"
    )


@_artifacts
def test_dataset_source_is_reported_not_assumed():
    """Task 7 must report the same synthetic-vs-real status Task 6 does."""
    metrics = json.loads(DL_METRICS_JSON.read_text())
    assert "dataset_source" in metrics
    assert metrics["dataset_source"].get("source") in {"real_fastf1", "synthetic", "unknown"}


@_artifacts
def test_undefined_metrics_are_null_not_zero():
    """A metric that is mathematically undefined on a split must be recorded as
    null with a reason, never as a plausible-looking 0.0."""
    metrics = json.loads(DL_METRICS_JSON.read_text())
    clf = metrics["models"]["target_pit_next_lap"]["test_metrics"]
    if clf.get("roc_auc") is None:
        assert clf.get("undefined_reason"), "undefined metric recorded without a reason"


@_artifacts
def test_artifacts_exist_helper_agrees_with_the_filesystem():
    assert dl_pipeline.artifacts_exist() is True


# --------------------------------------------------------------------------
# Registry survival (TODO.md — "A non-hermetic test run silently deleted
# Task 7's registry rows")
# --------------------------------------------------------------------------
def test_task6_retrain_preserves_task7_registry_entries(tmp_path):
    """Task 7 extends the shared registry rather than keeping a parallel one.
    A Task 6 retrain must not delete its rows — that is exactly what happened
    before ``write_registry`` learned to preserve foreign families, leaving
    /api/dl/models returning 404 while the .keras files sat on disk.
    """
    import json

    from app.intelligence.ml.registry import ModelRegistryEntry, load_registry, write_registry

    registry_path = tmp_path / "model_registry.json"

    # A registry holding one classical and one deep entry, as build_all produces.
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({
        "generated_at": "2026-01-01T00:00:00+00:00",
        "models": [
            {"model_name": "old_classical", "family": "classical", "target": "target_laptime"},
            {"model_name": "dnn_mlp", "family": "deep", "target": "target_laptime",
             "artifact": "models/dl/target_laptime.keras"},
            {"model_name": "dnn_mlp", "family": "deep", "target": "target_pit_next_lap",
             "artifact": "models/dl/target_pit_next_lap.keras"},
        ],
    }, indent=2))

    # Simulate a Task 6 retrain writing an entirely new set of classical entries.
    fresh = ModelRegistryEntry(
        model_name="random_forest", task="target_laptime", target="target_laptime",
        features=["a"], validation={}, metrics={}, artifact="models/laptime/random_forest.joblib",
        hyperparameters={}, random_state=42, dataset="test", training_rows=10,
        test_rows=2, cv_folds=4, is_selected_best=True, synthetic_data_warning=False,
    )
    write_registry([fresh], registry_path)

    after = load_registry(registry_path)
    families = [m.get("family", "classical") for m in after["models"]]
    deep = [m for m in after["models"] if m.get("family") == "deep"]

    assert "deep" in families, "a Task 6 retrain deleted Task 7's registry rows"
    assert len(deep) == 2, f"expected both deep entries preserved, got {len(deep)}"
    assert {m["target"] for m in deep} == {"target_laptime", "target_pit_next_lap"}
    # The stale classical entry must be gone — only foreign families are kept.
    assert "old_classical" not in {m["model_name"] for m in after["models"]}


def test_restore_registry_entries_rebuilds_from_committed_artifacts():
    """The recovery path must work without retraining, since that is the whole
    point of it existing."""
    import json

    from app.core.paths import ML_MODEL_REGISTRY_JSON

    if not ML_MODEL_REGISTRY_JSON.exists() or not DL_METRICS_JSON.exists():
        import pytest as _pytest

        _pytest.skip("Tasks 6/7 not built")

    before = json.loads(ML_MODEL_REGISTRY_JSON.read_text())
    n_deep_before = sum(1 for m in before["models"] if m.get("family") == "deep")

    restored = dl_pipeline.restore_registry_entries()
    after = json.loads(ML_MODEL_REGISTRY_JSON.read_text())
    n_deep_after = sum(1 for m in after["models"] if m.get("family") == "deep")

    assert restored == n_deep_after == n_deep_before, "restore changed the deep entry count"
