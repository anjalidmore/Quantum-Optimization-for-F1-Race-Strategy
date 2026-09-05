"""
Unit tests for the f1dl deep-learning package (Phase 3, Task 7).

Run with:  pytest -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from f1dl import baselines, contract, evaluation, models, persistence, splits, training, tuning

OUTPUTS = _ROOT / "outputs"


# --------------------------------------------------------------------------
# Contract gate
# --------------------------------------------------------------------------
def test_contract_loads_and_validates():
    ds = contract.load_and_validate()
    assert len(ds.frame) > 0
    assert "target_laptime" in ds.frame.columns
    assert "target_pit_next_lap" in ds.frame.columns


def test_no_leakage_columns_in_feature_matrix():
    """The same leakage check Task 6 performs — sector times and speed traps
    must never reach a model, deep or classical."""
    ds = contract.load_and_validate()
    present = [c for c in ds.contract.leakage_columns if c in ds.frame.columns]
    assert present == [], f"Leakage columns leaked into the matrix: {present}"


def test_no_target_is_its_own_predictor():
    ds = contract.load_and_validate()
    targets = set(ds.contract.target_names)
    for t in ("target_laptime", "target_pit_next_lap"):
        assert not (set(ds.contract.selected_features(t)) & targets)


def test_contract_error_on_missing_files(tmp_path):
    with pytest.raises(contract.DataContractError):
        contract.load_and_validate(practical05_outputs=tmp_path)


def test_identity_features_are_detected():
    """Task 8's fairness assessment depends on this classification being right."""
    ds = contract.load_and_validate()
    ident = ds.contract.identity_features("target_laptime")
    assert all(f.startswith(("driver_", "team_")) for f in ident)


# --------------------------------------------------------------------------
# Splits — time-ordering must hold
# --------------------------------------------------------------------------
def _panel(n_laps=40, n_drivers=6):
    rows = [{"LapNumber": lap, "f": float(lap + d), "y": float(lap)}
            for lap in range(1, n_laps + 1) for d in range(n_drivers)]
    return pd.DataFrame(rows)


def test_holdout_is_chronological_and_disjoint():
    df = _panel()
    h = splits.chronological_holdout(df, test_fraction=0.2)
    assert max(h.dev_laps) < min(h.test_laps)
    assert set(h.dev_index).isdisjoint(set(h.test_index))
    assert len(h.dev_index) + len(h.test_index) == len(df)


def test_expanding_window_never_trains_on_the_future():
    df = _panel()
    for fold in splits.expanding_window_folds(df, n_folds=4):
        assert max(fold.train_laps) < min(fold.val_laps), (
            "an expanding-window fold trained on laps at or after its validation block"
        )


def test_whole_laps_stay_in_one_fold():
    df = _panel()
    for fold in splits.expanding_window_folds(df, n_folds=4):
        assert not (set(fold.train_laps) & set(fold.val_laps))


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def test_regression_head_is_linear_and_classification_head_is_sigmoid():
    reg = models.build_regression_mlp(n_features=10)
    clf = models.build_classification_mlp(n_features=10)
    assert reg.layers[-1].activation.__name__ == "linear"
    assert clf.layers[-1].activation.__name__ == "sigmoid"


def test_architecture_summary_is_json_serialisable():
    s = models.architecture_summary(models.build_regression_mlp(n_features=8))
    json.dumps(s)
    assert s["total_parameters"] > 0


# --------------------------------------------------------------------------
# Training — the leakage-safe scaling contract
# --------------------------------------------------------------------------
def test_scaler_is_fit_on_training_rows_only():
    """Validation statistics must not influence the transform. If the scaler
    had seen the validation rows, the training rows would not be exactly
    zero-mean."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (50, 3))
    Xva = rng.normal(50, 1, (20, 3))       # wildly different distribution
    mask = np.array([True, True, True])
    Xt, Xv, _ = training.scale_fit_transform(Xtr, Xva, mask)
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-5)
    assert np.abs(Xv.mean()) > 10, "validation rows were used to fit the scaler"


def test_binary_columns_are_not_scaled():
    Xtr = np.column_stack([np.random.normal(0, 1, 40), np.random.randint(0, 2, 40)])
    mask = np.array([True, False])
    Xt, _, _ = training.scale_fit_transform(Xtr, Xtr, mask)
    assert set(np.unique(Xt[:, 1])) <= {0.0, 1.0}


def test_balanced_class_weight_upweights_the_minority():
    y = np.array([0] * 95 + [1] * 5)
    w = training.balanced_class_weight(y)
    assert w[1] > w[0]


def test_model_trains_without_error_and_history_is_recorded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype("float32")
    y = (X[:, 0] * 2 + rng.normal(0, 0.1, 120)).astype("float32")
    mask = np.ones(5, dtype=bool)
    fit = training.fit_fold(
        models.build_regression_mlp, X[:90], y[:90], X[90:], y[90:], mask,
        hidden_units=(8,), dropout=0.1, learning_rate=1e-2, batch_size=16,
        max_epochs=12, patience=5, scale_target=True,
    )
    assert fit.epochs_run > 0
    assert "loss" in fit.history and "val_loss" in fit.history
    assert fit.y_scaler is not None


def test_target_scaling_round_trips_to_real_units():
    """Predictions must come back in seconds, not standardised units — this is
    the bug that made the first run report a 29-second MAE."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4)).astype("float32")
    y = (91.0 + rng.normal(0, 0.5, 120)).astype("float32")   # realistic lap times
    mask = np.ones(4, dtype=bool)
    fit = training.fit_fold(
        models.build_regression_mlp, X[:90], y[:90], X[90:], y[90:], mask,
        hidden_units=(8,), dropout=0.0, learning_rate=1e-2, batch_size=16,
        max_epochs=40, patience=10, scale_target=True,
    )
    pred = training.predict(fit.model, fit.scaler, X[90:], mask, fit.y_scaler)
    assert 85 < float(np.mean(pred)) < 97, f"predictions not in lap-time units: {np.mean(pred)}"


def test_training_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4)).astype("float32")
    y = rng.normal(size=80).astype("float32")
    mask = np.ones(4, dtype=bool)
    kw = dict(hidden_units=(8,), dropout=0.1, learning_rate=1e-2,
              batch_size=16, max_epochs=8, patience=4)
    a = training.fit_fold(models.build_regression_mlp, X[:60], y[:60], X[60:], y[60:], mask, **kw)
    b = training.fit_fold(models.build_regression_mlp, X[:60], y[:60], X[60:], y[60:], mask, **kw)
    pa = training.predict(a.model, a.scaler, X[60:], mask)
    pb = training.predict(b.model, b.scaler, X[60:], mask)
    assert np.allclose(pa, pb), "same seed produced different predictions"


# --------------------------------------------------------------------------
# Evaluation honesty
# --------------------------------------------------------------------------
def test_single_class_split_reports_undefined_not_a_number():
    m = evaluation.classification_metrics(
        np.zeros(20, dtype=int), np.zeros(20, dtype=int), y_proba=np.full(20, 0.1)
    )
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["undefined_reason"]


def test_search_space_is_fully_enumerated():
    space = tuning.REGRESSION_SPACE
    n = len(space.hidden_units) * len(space.dropout) * len(space.learning_rate) * len(space.batch_size)
    assert len(space.combinations()) == n == space.to_metadata()["total_combinations"]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_saved_model_reloads_and_predicts(tmp_path):
    m = models.build_regression_mlp(n_features=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.random.normal(size=(20, 4)))
    saved = persistence.save(m, sc, ["a", "b", "c", "d"], np.ones(4, dtype=bool), "t", tmp_path)
    assert saved.model_path.exists() and saved.model_path.suffix == ".keras"
    m2, sc2, ys2, spec = persistence.load("t", tmp_path)
    assert spec["features"] == ["a", "b", "c", "d"]
    assert m2.predict(np.random.normal(size=(2, 4)).astype("float32"), verbose=0).shape == (2, 1)


# --------------------------------------------------------------------------
# Committed artifacts — no fabricated results
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (OUTPUTS / "metadata" / "dl_model_registry.json").exists(),
                    reason="run practical06.py first")
def test_registry_metrics_match_the_evaluation_report():
    reg = json.loads((OUTPUTS / "metadata" / "dl_model_registry.json").read_text())
    report = (OUTPUTS / "reports" / "dl_evaluation_report.md").read_text()
    for entry in reg["models"]:
        if entry["target"] == "target_laptime":
            mae = entry["test_metrics"]["mae"]
            assert f"{mae:.4f}" in report, "registry MAE does not appear in the report"


@pytest.mark.skipif(not (OUTPUTS / "history").exists(), reason="run practical06.py first")
def test_training_history_is_real_and_monotonic_in_length():
    for f in (OUTPUTS / "history").glob("*_history.json"):
        h = json.loads(f.read_text())
        assert h["epochs_run"] == len(h["history"]["loss"]) > 0
        assert h["best_epoch"] <= h["epochs_run"]
