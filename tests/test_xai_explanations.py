"""
Task 8 — explainable AI tests.

Two kinds of check: the explainers behave correctly on a tiny problem whose
right answer is known, and the committed artifacts are internally consistent
and traceable — including a recomputation of every trust score from the inputs
the artifact itself records.
"""
from __future__ import annotations

import json
import warnings

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()
warnings.filterwarnings("ignore")

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.core.paths import ML_REPORTS_DIR, XAI_RESULTS_JSON
from app.intelligence.xai import (
    counterfactual, fairness, importance, lime_analysis,
    narrative, shap_analysis, trust,
)
from app.intelligence.xai import pipeline as xai_pipeline

_artifacts = pytest.mark.skipif(
    not XAI_RESULTS_JSON.exists(), reason="Task 8 not built; run scripts/build_all.py"
)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.05, 200)
    names = ["signal", "weak_signal", "noise_a", "noise_b"]
    return X, y, names, RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)


# --------------------------------------------------------------------------
# Explainers behave correctly where the answer is known
# --------------------------------------------------------------------------
def test_permutation_importance_finds_the_real_driver(toy):
    X, y, names, model = toy
    rows = importance.permutation_importance(model.predict, X, y, names, "regression", n_repeats=5)
    assert rows[0]["feature"] == "signal"
    assert {rows[-1]["feature"], rows[-2]["feature"]} == {"noise_a", "noise_b"}


def test_permutation_importance_reports_undefined_rather_than_zero(toy):
    X, y, names, model = toy
    rows = importance.permutation_importance(
        model.predict, X, np.zeros(len(y), dtype=int), names, "classification", n_repeats=2)
    assert all(r["importance"] is None and r["undefined_reason"] for r in rows)


def test_tree_shap_is_exact_and_kernel_shap_is_labelled_approximate(toy):
    X, y, names, model = toy
    tree = shap_analysis.tree_shap(model, X[:50], names, "regression")
    kern = shap_analysis.kernel_shap(model.predict, X[:60], X[:3], names, nsamples=40, k=8)
    assert tree["exact"] is True and shap_analysis.global_ranking(tree)[0]["feature"] == "signal"
    assert kern["exact"] is False and "sampling noise" in kern["note"]


def test_shap_row_explanation_is_ordered_by_magnitude(toy):
    X, y, names, model = toy
    res = shap_analysis.tree_shap(model, X[:20], names, "regression")
    mags = [abs(r["shap_value"]) for r in shap_analysis.explain_row(res, 0, top_n=4)]
    assert mags == sorted(mags, reverse=True)


def test_lime_reports_surrogate_quality_and_is_reproducible(toy):
    X, y, names, model = toy
    out = []
    for _ in range(2):
        ex = lime_analysis.build_explainer(X, names, "regression", seed=7)
        r = lime_analysis.explain_row(ex, X[0], model.predict, "regression",
                                      num_features=4, num_samples=500)
        assert 0.0 <= r["local_r2"] <= 1.0
        out.append([c["weight"] for c in r["contributions"]])
    assert np.allclose(out[0], out[1]), "same seed produced different LIME weights"


def test_lime_conditions_map_back_to_feature_names(toy):
    X, y, names, model = toy
    ex = lime_analysis.build_explainer(X, names, "regression")
    r = lime_analysis.explain_row(ex, X[0], model.predict, "regression",
                                  num_features=4, num_samples=500)
    top = lime_analysis.top_features(r, names, top_k=3)
    assert top and all(f in names for f in top)


# --------------------------------------------------------------------------
# Counterfactuals
# --------------------------------------------------------------------------
def test_perturbation_scan_locates_a_real_crossing():
    cf = counterfactual.perturbation_scan(
        lambda rows: np.asarray(rows)[:, 0], np.array([0.0, 5.0]), 0, "f0",
        -10.0, 10.0, threshold=3.0, steps=200)
    assert cf["reachable"] and abs(cf["crossing_value"] - 3.0) < 0.2
    assert cf["direction"] == "increase"


def test_perturbation_scan_reports_unreachable_with_the_range_searched():
    cf = counterfactual.perturbation_scan(
        lambda rows: np.zeros(len(rows)), np.array([0.0]), 0, "f0", -1.0, 1.0, 99.0, steps=20)
    assert cf["reachable"] is False and cf["crossing_value"] is None
    assert cf["searched_range"] == [-1.0, 1.0]


def test_perturbation_scan_holds_other_features_fixed():
    seen = []

    def predict(rows):
        seen.append(np.asarray(rows)[:, 1].copy())
        return np.asarray(rows)[:, 0]

    counterfactual.perturbation_scan(predict, np.array([0.0, 7.5]), 0, "f0", -1.0, 1.0, 0.0, steps=10)
    assert np.allclose(seen[0], 7.5)


def test_dice_time_budget_is_enforced():
    import time

    with pytest.raises(counterfactual._SearchTimeout):
        with counterfactual._time_budget(1):
            time.sleep(3)


def test_dice_reports_single_class_data_rather_than_hanging():
    X = np.random.default_rng(0).normal(size=(50, 3))
    res = counterfactual.dice_counterfactuals(
        RandomForestClassifier(n_estimators=5, random_state=0).fit(X, np.zeros(50, dtype=int)),
        X, np.zeros(50, dtype=int), X[0], ["a", "b", "c"])
    assert res["available"] is False and "single class" in res["reason"]


# --------------------------------------------------------------------------
# Trust score
# --------------------------------------------------------------------------
def test_trust_is_one_when_everything_agrees():
    t = trust.compute(task="classification", dnn_prediction=1.0, classical_prediction=1.0,
                      shap_top=["a", "b", "c"], lime_top=["a", "b", "c"])
    assert t["trust_score"] == pytest.approx(1.0) and t["band"]["label"] == "HIGH"


def test_boundary_prediction_with_contradicting_models_is_do_not_act():
    t = trust.compute(task="classification", dnn_prediction=0.5, classical_prediction=1.0,
                      shap_top=["a"], lime_top=["z"])
    assert t["band"]["label"] == "DO NOT ACT"


def test_trust_components_stay_within_the_unit_interval():
    for a, b in [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.73, 0.11)]:
        t = trust.compute(task="classification", dnn_prediction=a, classical_prediction=b,
                          shap_top=["a", "b"], lime_top=["b", "c"])
        assert all(0.0 <= v <= 1.0 for v in t["components"].values())
        assert 0.0 <= t["trust_score"] <= 1.0


def test_weights_sum_to_one():
    assert sum(trust.WEIGHTS.values()) == pytest.approx(1.0)


def test_regression_trust_requires_a_target_scale():
    with pytest.raises(ValueError):
        trust.compute(task="regression", dnn_prediction=91.0, classical_prediction=91.1,
                      shap_top=["a"], lime_top=["a"], target_std=None)


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------
def test_fairness_flags_identity_domination():
    a = fairness.assess(
        [{"feature": "driver_ham", "mean_abs_shap": 0.80},
         {"feature": "tyre_life", "mean_abs_shap": 0.10},
         {"feature": "race_progress", "mean_abs_shap": 0.10}],
        ["driver_ham"])
    assert a["concentration_ratio"] > 2.0
    assert "predicting *who* rather than *what*" in a["reading"]


def test_fairness_calls_a_healthy_model_healthy():
    a = fairness.assess(
        [{"feature": "tyre_life", "mean_abs_shap": 0.60},
         {"feature": "race_progress", "mean_abs_shap": 0.35},
         {"feature": "driver_ham", "mean_abs_shap": 0.05}],
        ["driver_ham"])
    assert a["concentration_ratio"] < 1.0 and "desired outcome" in a["reading"]


def test_fairness_shares_sum_to_one():
    a = fairness.assess(
        [{"feature": "driver_ham", "mean_abs_shap": 0.3},
         {"feature": "tyre_life", "mean_abs_shap": 0.7}], ["driver_ham"])
    assert a["identity_attribution_share"] + a["race_state_attribution_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Narrative — race-engineer language only
# --------------------------------------------------------------------------
def test_pit_sentence_states_the_recommendation_and_its_factors():
    s = narrative.pit_decision_sentence(
        0.82, [{"feature": "tyre_life", "shap_value": 0.4, "direction": "increases"}],
        {"tyre_life": 18.0}, "HIGH")
    assert "Recommend PITTING" in s and "tyre age" in s.lower() and "HIGH" in s


def test_low_probability_does_not_round_to_zero_percent():
    s = narrative.pit_decision_sentence(
        0.0004, [{"feature": "tyre_life", "shap_value": -0.4, "direction": "decreases"}],
        {"tyre_life": 3.0}, "HIGH")
    assert "<1%" in s, "a 0.04% probability was rounded to 0%, overstating certainty"


def test_identity_features_are_humanised_as_identity():
    assert "driver identity" in narrative.humanise("driver_ham")
    assert "team identity" in narrative.humanise("team_red_bull_racing")


def test_unknown_feature_is_not_silently_dropped():
    assert narrative.humanise("brand_new_feature_v2") == "brand new feature v2"


def test_no_clinical_vocabulary_in_generated_explanations():
    """The reference task spec is a clinical decision-support system; none of
    its vocabulary belongs in this project's output."""
    banned = ("patient", "clinical", "disease", "diagnosis", "clinician", "medical")
    s = narrative.pit_decision_sentence(
        0.9, [{"feature": "tyre_life", "shap_value": 0.5, "direction": "increases"}],
        {"tyre_life": 22.0}, "HIGH").lower()
    assert not any(w in s for w in banned)
    for v in narrative.GLOSSARY.values():
        assert not any(w in v.lower() for w in banned)


# --------------------------------------------------------------------------
# Committed artifacts
# --------------------------------------------------------------------------
@_artifacts
def test_every_target_has_real_explanations():
    data = xai_pipeline.load_results()["targets"]
    assert set(data) == {"target_laptime", "target_pit_next_lap"}
    for target, r in data.items():
        assert r["examples"], f"{target} has no explained rows"
        assert r["shap"]["dnn"]["ranking"], f"{target} has no SHAP ranking"
        for label, ex in r["examples"].items():
            assert ex["shap_dnn"], f"{target}/{label} has no attributions"
            assert ex["narrative"], f"{target}/{label} has no strategic explanation"


@_artifacts
def test_explained_rows_are_distinct():
    for target, r in xai_pipeline.load_results()["targets"].items():
        idx = [ex["row_index"] for ex in r["examples"].values()]
        assert len(idx) == len(set(idx)), f"{target} explained the same row twice"


@_artifacts
def test_trust_scores_recompute_from_their_own_recorded_inputs():
    """The strongest no-fabrication check available: every committed trust
    score must be reproducible from the inputs the artifact itself stores."""
    for target, r in xai_pipeline.load_results()["targets"].items():
        for label, ex in r["examples"].items():
            t = ex["trust"]
            again = trust.compute(
                task=r["task"],
                dnn_prediction=t["inputs"]["dnn_prediction"],
                classical_prediction=t["inputs"]["classical_prediction"],
                shap_top=t["inputs"]["shap_top3"],
                lime_top=t["inputs"]["lime_top3"],
                target_std=t["inputs"]["target_std"])
            assert again["trust_score"] == pytest.approx(t["trust_score"]), f"{target}/{label}"


@_artifacts
def test_fairness_share_recomputes_from_the_committed_shap_ranking():
    for target, r in xai_pipeline.load_results()["targets"].items():
        again = fairness.assess(r["shap"]["dnn"]["ranking"], r["identity_features"])
        assert again["identity_attribution_share"] == pytest.approx(
            r["fairness"]["identity_attribution_share"])


@_artifacts
def test_reports_exist_and_are_non_trivial():
    for name in ("xai_shap_report.md", "xai_lime_report.md", "xai_counterfactual_report.md",
                 "xai_trust_score_report.md", "xai_fairness_report.md",
                 "xai_explainability_dashboard.md"):
        p = ML_REPORTS_DIR / name
        assert p.exists(), f"missing deliverable: {name}"
        assert len(p.read_text()) > 500, f"{name} is suspiciously short"


@_artifacts
def test_dataset_source_is_reported_in_task8_too():
    data = xai_pipeline.load_results()
    assert data["dataset_source"].get("source") in {"real_fastf1", "synthetic", "unknown"}
