"""
Regression tests for the defects found while implementing the Task 7 / Task 8
specification of 2026-09-21. Each test pins one of them so it cannot return
silently:

1. ``model.fit(class_weight=...)`` does not weight the loss on this Keras/torch
   installation, so the classifier was effectively unweighted.
2. The final Task 7 network early-stopped on rows that were also in its
   training set, so its "validation" curve was a training curve.
3. The tyre-age counterfactual moved ``tyre_life`` alone, producing inputs no
   real lap can have.
4. The trust score measured confidence from 0.5, not from the threshold the
   model actually decides at.
5. No F1-specific performance stratification existed.
6. ``/api/dl/predict/pit`` decided at 0.5 instead of the tuned threshold.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.paths import DL_METRICS_JSON, TASK5_FEATURE_METADATA_JSON, TASK5_FEATURES_CSV, XAI_RESULTS_JSON
from app.intelligence.dl import models, training
from app.intelligence.xai import counterfactual, stratification, trust

client = TestClient(app)
_dl = pytest.mark.skipif(not DL_METRICS_JSON.exists(), reason="Task 7 not built")
_xai = pytest.mark.skipif(not XAI_RESULTS_JSON.exists(), reason="Task 8 not built")


# ---------------------------------------------------------------------------
# 1. Class weighting must actually change the model
# ---------------------------------------------------------------------------
def test_class_weighting_inside_the_loss_moves_the_predictions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8)).astype("float32")
    y = (rng.random(600) < 0.05).astype("float32")

    def mean_p(cw):
        training.set_seeds(1)
        m = models.build_classification_mlp(8, (16,), dropout=0.0, l2=0.0, learning_rate=1e-2, class_weight=cw)
        m.fit(X, y, epochs=30, batch_size=32, verbose=0)
        return float(m.predict(X, verbose=0).mean())

    unweighted, weighted = mean_p(None), mean_p({0: 0.5, 1: 10.0})
    # Up-weighting the rare class 20:1 must push its probabilities up materially
    # (fit(class_weight=...) moved this from 0.039 to only 0.050 when measured).
    assert weighted > unweighted + 0.10, (unweighted, weighted)


def test_training_never_passes_class_weight_to_fit():
    import inspect
    import re

    src = inspect.getsource(training.fit_fold)
    call = re.search(r"model\.fit\((.*?)verbose=verbose", src, re.S)
    assert call, "could not locate the model.fit call"
    assert "class_weight" not in call.group(1), "class weights must go into the loss, not into fit()"


# ---------------------------------------------------------------------------
# 2. The final fit is time-aware and its validation rows are unseen
# ---------------------------------------------------------------------------
@_dl
def test_final_fit_validates_on_later_unseen_laps_and_tests_on_the_last_laps():
    report = json.loads(DL_METRICS_JSON.read_text())
    for target, m in report["models"].items():
        s = m["final_fit_split"]
        assert max(s["train_laps"]) < min(s["validation_laps"]), f"{target}: validation laps overlap training"
        assert max(s["validation_laps"]) < min(s["test_laps"]), f"{target}: test laps are not the latest"
        assert set(s["train_laps"]).isdisjoint(s["validation_laps"])


@_dl
def test_saved_models_are_h5_and_reload_verified():
    from app.core.paths import ARTIFACTS_DIR

    report = json.loads(DL_METRICS_JSON.read_text())
    for target, m in report["models"].items():
        mf = m["model_file"]
        assert mf["path"].endswith("f1_dnn_model.h5")
        assert mf["reload_verified"] is True
        assert (ARTIFACTS_DIR.parent / mf["path"]).exists()


@_dl
def test_hyperparameter_report_covers_the_specified_values_and_selects_one_per_target():
    from app.core.paths import DEEP_LEARNING_DIR

    hp = pd.read_csv(DEEP_LEARNING_DIR / "hyperparameter_report.csv")
    for target, sub in hp.groupby("Target"):
        assert (sub.Selected == "YES").sum() == 1, f"{target}: exactly one selected configuration"
        assert {0.001, 0.0005, 0.0001} <= set(sub["Learning Rate"].round(6))
        assert {16, 32, 64} <= set(sub["Batch Size"])
        assert {0.2, 0.3, 0.4, 0.5} <= set(sub["Dropout"].round(2))
        assert {"Adam", "RMSprop"} <= set(sub["Optimizer"])
    assert set(hp[hp.Target == "target_pit_next_lap"]["Class Weighting"]) >= {"none", "balanced (inside the loss)"}


# ---------------------------------------------------------------------------
# 3. Counterfactuals must stay physically possible
# ---------------------------------------------------------------------------
def _real_laps(target: str):
    meta = json.loads(TASK5_FEATURE_METADATA_JSON.read_text())
    feats = meta["selected_features"][target]
    return feats, pd.read_csv(TASK5_FEATURES_CSV)[feats].to_numpy(float)


@pytest.mark.parametrize("target", ["target_laptime", "target_pit_next_lap"])
def test_consistent_tyre_rows_reproduce_every_real_lap_at_its_own_tyre_age(target):
    feats, X = _real_laps(target)
    i_life = feats.index("tyre_life")
    for row in X:
        rebuilt, _, _ = counterfactual.consistent_tyre_age_rows(row, feats, np.array([row[i_life]]))
        assert np.allclose(rebuilt[0], row, atol=1e-4)


def test_changing_tyre_age_recomputes_the_interaction_terms():
    feats, X = _real_laps("target_laptime")
    row = X[0]
    rows, recomputed, held = counterfactual.consistent_tyre_age_rows(row, feats, np.array([20.0]))
    r = rows[0]
    i = {f: k for k, f in enumerate(feats)}
    assert r[i["tyre_life"]] == 20.0
    assert np.isclose(r[i["tyrelife_x_soft"]], 20.0 * row[i["compound_soft"]])
    assert np.isclose(r[i["tyrelife_x_medium"]], 20.0 * row[i["compound_medium"]])
    assert "is_fresh_tyre" in held and "compound_soft" in held


# ---------------------------------------------------------------------------
# 4. Trust score
# ---------------------------------------------------------------------------
def test_confidence_is_measured_from_the_tuned_threshold():
    # At the model's own threshold, confidence must be zero - not 2*|0.1-0.5| = 0.8.
    assert trust.confidence_from_threshold(0.1, 0.1) == 0.0
    assert trust.confidence_from_threshold(1.0, 0.1) == pytest.approx(1.0)
    assert trust.confidence_from_threshold(0.0, 0.1) == pytest.approx(1.0)


def test_missing_components_are_renormalised_not_scored_as_zero():
    partial = trust.compute(task="classification", dnn_prediction=0.9, classical_prediction=0.9,
                            threshold=0.1, input_validity_share=1.0)
    assert partial["renormalised"] and "explanation_stability" not in partial["components"]
    w, c = trust.WEIGHTS, partial["components"]
    present = [k for k in c]
    weighted_mean = sum(w[k] * c[k] for k in present) / sum(w[k] for k in present)
    zero_filled = sum(w[k] * c[k] for k in present)          # what scoring the gap as 0 would give
    assert partial["trust_score"] == pytest.approx(weighted_mean, abs=1e-4)
    assert partial["trust_score"] > zero_filled + 0.05


def test_input_validity_flags_values_outside_the_training_range():
    X_train = np.random.default_rng(0).normal(size=(500, 4))
    assert trust.input_validity(np.zeros(4), X_train) == 1.0
    assert trust.input_validity(np.array([0, 0, 50, 50]), X_train) == 0.5


# ---------------------------------------------------------------------------
# 5. F1-specific performance stratification
# ---------------------------------------------------------------------------
def test_stratification_reports_groups_and_flags_small_samples():
    ids = pd.DataFrame({"Driver": ["A"] * 40 + ["B"] * 5, "Team": ["T"] * 45, "Compound": ["SOFT"] * 45})
    y = np.r_[np.zeros(35), np.ones(5), np.zeros(5)]
    p = np.r_[np.full(35, 0.05), np.full(5, 0.9), np.full(5, 0.05)]
    s = stratification.stratify("classification", ids, y, p, threshold=0.5)
    overall = s[s.group_type == "overall"].iloc[0]
    assert overall.n_laps == 45 and overall.recall == 1.0
    b = s[(s.group_type == "Driver") & (s.group == "B")].iloc[0]
    assert b.sample_note.startswith("INSUFFICIENT")


@_xai
def test_stratification_endpoint_serves_driver_team_and_compound():
    body = client.get("/api/xai/stratification").json()
    for target, s in body.items():
        kinds = {r["group_type"] for r in s["rows"]}
        assert {"overall", "Driver", "Team", "Compound"} <= kinds, target


@_xai
def test_lap_inspector_endpoint_returns_a_full_scenario():
    laps = client.get("/api/xai/laps", params={"target": "target_pit_next_lap"}).json()["laps"]
    assert laps
    body = client.get("/api/xai/lap", params={"target": "target_pit_next_lap", "row_index": 0}).json()
    assert set(body) >= {"lap", "race_state", "shap_factors", "counterfactual", "decision_threshold"}
    cf = body["counterfactual"]
    assert cf.get("derived_features_recomputed"), "the live counterfactual must recompute tyre-derived features"
    # SHAP additivity: base + contributions == the model's prediction for this lap
    total = body["shap_base_value"] + sum(f["shap_value"] for f in body["shap_factors"])
    assert total == pytest.approx(body["lap"]["dnn_prediction"], abs=2e-3)


@_xai
def test_xai_deliverables_exist():
    from app.core.paths import XAI_DIR

    for rel in ("feature_importance.png", "counterfactual_analysis.csv", "trust_score_report.csv",
                "fairness_assessment.csv", "xai_metadata.json", "SHAP_Report.md", "LIME_Report.md"):
        assert (XAI_DIR / rel).exists(), rel
    assert any((XAI_DIR / "shap").glob("*_shap_summary.png"))
    assert any((XAI_DIR / "lime").glob("*_lime.png"))


# ---------------------------------------------------------------------------
# 6. The API decides pit/no-pit at the tuned threshold
# ---------------------------------------------------------------------------
@_dl
def test_dl_pit_predicted_class_uses_the_tuned_threshold():
    from app.api.routers.dl import _pit_threshold

    report = json.loads(DL_METRICS_JSON.read_text())
    tuned = report["models"]["target_pit_next_lap"]["threshold"]["threshold"]
    assert _pit_threshold() == pytest.approx(tuned)


@_xai
def test_xai_metadata_is_strict_json_so_every_endpoint_can_serve_it():
    # json.dumps writes NaN for an undefined metric (recall on a compound with no
    # pit laps); the API then fails with a 500. Undefined must be stored as null.
    json.loads(XAI_RESULTS_JSON.read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert client.get("/api/xai/stratification").status_code == 200
