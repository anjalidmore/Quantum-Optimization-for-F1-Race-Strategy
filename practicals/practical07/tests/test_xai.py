"""
Unit tests for the f1xai explainability package (Phase 3, Task 8).

Run with:  pytest -q
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
warnings.filterwarnings("ignore")

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from f1xai import (
    counterfactual, fairness, importance, lime_analysis,
    narrative, shap_analysis, trust,
)

OUTPUTS = _ROOT / "outputs"


# --------------------------------------------------------------------------
# Fixtures — a tiny, fully controlled problem where the right answer is known
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    # feature 0 drives the target; features 2 and 3 are pure noise
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.05, 200)
    names = ["signal", "weak_signal", "noise_a", "noise_b"]
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    return X, y, names, model


@pytest.fixture(scope="module")
def toy_classification():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (2.0 * X[:, 0] + rng.normal(0, 0.2, 200) > 0).astype(int)
    names = ["signal", "weak_signal", "noise_a", "noise_b"]
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    return X, y, names, model


# --------------------------------------------------------------------------
# Permutation importance
# --------------------------------------------------------------------------
def test_permutation_importance_finds_the_real_driver(toy_regression):
    X, y, names, model = toy_regression
    rows = importance.permutation_importance(model.predict, X, y, names, "regression", n_repeats=5)
    assert rows[0]["feature"] == "signal", f"ranked {rows[0]['feature']} above the true driver"
    assert rows[0]["importance"] > 0


def test_permutation_importance_ranks_noise_last(toy_regression):
    X, y, names, model = toy_regression
    rows = importance.permutation_importance(model.predict, X, y, names, "regression", n_repeats=5)
    assert {rows[-1]["feature"], rows[-2]["feature"]} == {"noise_a", "noise_b"}


def test_permutation_importance_reports_undefined_not_zero_on_single_class(toy_classification):
    """A split with one class makes ROC-AUC undefined. The function must say
    so rather than emit a plausible-looking 0.0."""
    X, y, names, model = toy_classification
    single = np.zeros(len(y), dtype=int)
    rows = importance.permutation_importance(
        lambda r: model.predict_proba(r)[:, 1], X, single, names, "classification", n_repeats=2)
    assert all(r["importance"] is None for r in rows)
    assert all(r["undefined_reason"] for r in rows)


def test_compare_importance_aligns_both_rankings(toy_regression):
    X, y, names, model = toy_regression
    a = importance.permutation_importance(model.predict, X, y, names, "regression", n_repeats=3)
    b = importance.permutation_importance(model.predict, X, y, names, "regression", n_repeats=3)
    cmp = importance.compare_importance(a, b)
    assert len(cmp) == len(names)
    assert all(r["rank_gap"] is not None for r in cmp)


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def test_tree_shap_is_marked_exact_and_ranks_the_driver_first(toy_regression):
    X, y, names, model = toy_regression
    res = shap_analysis.tree_shap(model, X[:60], names, "regression")
    assert res["exact"] is True and res["explainer"] == "TreeExplainer"
    assert shap_analysis.global_ranking(res)[0]["feature"] == "signal"


def test_kernel_shap_is_marked_approximate_and_records_its_budget(toy_regression):
    X, y, names, model = toy_regression
    res = shap_analysis.kernel_shap(model.predict, X[:80], X[:3], names, nsamples=40, k=8)
    assert res["exact"] is False and res["explainer"] == "KernelExplainer"
    assert res["nsamples"] == 40 and res["background_k"] == 8
    assert "sampling noise" in res["note"]
    assert res["values"].shape == (3, len(names))


def test_explain_row_is_ordered_by_magnitude(toy_regression):
    X, y, names, model = toy_regression
    res = shap_analysis.tree_shap(model, X[:20], names, "regression")
    row = shap_analysis.explain_row(res, 0, top_n=4)
    mags = [abs(r["shap_value"]) for r in row]
    assert mags == sorted(mags, reverse=True)


def test_explain_row_direction_matches_sign(toy_regression):
    X, y, names, model = toy_regression
    res = shap_analysis.tree_shap(model, X[:20], names, "regression")
    for r in shap_analysis.explain_row(res, 0, top_n=4):
        assert r["direction"] == ("increases" if r["shap_value"] > 0 else "decreases")


def test_classification_shap_collapses_to_the_positive_class(toy_classification):
    X, y, names, model = toy_classification
    res = shap_analysis.tree_shap(model, X[:30], names, "classification")
    assert res["values"].shape == (30, len(names)), "per-class axis was not reduced"


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------
def test_lime_regression_runs_and_reports_surrogate_quality(toy_regression):
    X, y, names, model = toy_regression
    ex = lime_analysis.build_explainer(X, names, "regression")
    res = lime_analysis.explain_row(ex, X[0], model.predict, "regression",
                                    num_features=4, num_samples=500)
    assert res["contributions"]
    assert 0.0 <= res["local_r2"] <= 1.0, "local_r2 outside [0,1]"
    assert "not Shapley values" in res["note"]


def test_lime_classification_runs(toy_classification):
    X, y, names, model = toy_classification
    ex = lime_analysis.build_explainer(X, names, "classification")
    res = lime_analysis.explain_row(
        ex, X[0], lambda r: model.predict_proba(r)[:, 1], "classification",
        num_features=4, num_samples=500)
    assert res["contributions"]


def test_lime_top_features_maps_conditions_back_to_feature_names(toy_regression):
    X, y, names, model = toy_regression
    ex = lime_analysis.build_explainer(X, names, "regression")
    res = lime_analysis.explain_row(ex, X[0], model.predict, "regression",
                                    num_features=4, num_samples=500)
    top = lime_analysis.top_features(res, names, top_k=3)
    assert top and all(f in names for f in top), f"unmapped conditions: {top}"


def test_lime_is_reproducible_under_a_fixed_seed(toy_regression):
    X, y, names, model = toy_regression
    out = []
    for _ in range(2):
        ex = lime_analysis.build_explainer(X, names, "regression", seed=7)
        r = lime_analysis.explain_row(ex, X[0], model.predict, "regression",
                                      num_features=4, num_samples=500)
        out.append([c["weight"] for c in r["contributions"]])
    assert np.allclose(out[0], out[1]), "same seed produced different LIME weights"


# --------------------------------------------------------------------------
# Counterfactuals
# --------------------------------------------------------------------------
def test_perturbation_scan_finds_a_real_crossing():
    """A monotone model must have its threshold crossing located exactly."""
    def predict(rows):
        return np.asarray(rows)[:, 0] * 1.0

    row = np.array([0.0, 5.0])
    cf = counterfactual.perturbation_scan(predict, row, 0, "f0", -10.0, 10.0, threshold=3.0, steps=200)
    assert cf["reachable"]
    assert abs(cf["crossing_value"] - 3.0) < 0.2, cf["crossing_value"]
    assert cf["direction"] == "increase"


def test_perturbation_scan_reports_unreachable_honestly():
    def predict(rows):
        return np.zeros(len(rows))

    cf = counterfactual.perturbation_scan(predict, np.array([0.0]), 0, "f0", -1.0, 1.0,
                                           threshold=99.0, steps=20)
    assert cf["reachable"] is False
    assert cf["crossing_value"] is None
    assert "No crossing" in cf["note"]
    assert cf["searched_range"] == [-1.0, 1.0]


def test_perturbation_scan_holds_other_features_fixed():
    seen = []

    def predict(rows):
        seen.append(np.asarray(rows)[:, 1].copy())
        return np.asarray(rows)[:, 0]

    counterfactual.perturbation_scan(predict, np.array([0.0, 7.5]), 0, "f0", -1.0, 1.0, 0.0, steps=10)
    assert np.allclose(seen[0], 7.5), "a non-scanned feature was modified"


def test_dice_reports_single_class_training_data_rather_than_failing():
    X = np.random.default_rng(0).normal(size=(50, 3))
    res = counterfactual.dice_counterfactuals(
        RandomForestClassifier(n_estimators=5, random_state=0).fit(X, np.zeros(50, dtype=int)),
        X, np.zeros(50, dtype=int), X[0], ["a", "b", "c"])
    assert res["available"] is False
    assert "single class" in res["reason"]


def test_dice_time_budget_is_enforced():
    """An unbounded search that returns nothing is worse than a bounded one
    that says so."""
    import time
    with pytest.raises(counterfactual._SearchTimeout):
        with counterfactual._time_budget(1):
            time.sleep(3)


# --------------------------------------------------------------------------
# Trust score
# --------------------------------------------------------------------------
def test_trust_is_maximal_when_everything_agrees():
    t = trust.compute(task="classification", dnn_prediction=1.0, classical_prediction=1.0,
                      shap_top=["a", "b", "c"], lime_top=["a", "b", "c"])
    assert t["trust_score"] == pytest.approx(1.0)
    assert t["band"]["label"] == "HIGH"


def test_boundary_prediction_with_contradicting_models_scores_do_not_act():
    t = trust.compute(task="classification", dnn_prediction=0.5, classical_prediction=1.0,
                      shap_top=["a"], lime_top=["z"])
    assert t["trust_score"] < 0.25
    assert t["band"]["label"] == "DO NOT ACT"


def test_trust_components_are_each_bounded_to_unit_interval():
    for p_dnn, p_cls in [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.73, 0.11)]:
        t = trust.compute(task="classification", dnn_prediction=p_dnn, classical_prediction=p_cls,
                          shap_top=["a", "b"], lime_top=["b", "c"])
        for k, v in t["components"].items():
            assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"
        assert 0.0 <= t["trust_score"] <= 1.0


def test_weights_sum_to_one():
    assert sum(trust.WEIGHTS.values()) == pytest.approx(1.0)


def test_regression_trust_requires_target_std():
    with pytest.raises(ValueError):
        trust.compute(task="regression", dnn_prediction=91.0, classical_prediction=91.1,
                      shap_top=["a"], lime_top=["a"], target_std=None)


def test_jaccard_matches_hand_computation():
    assert trust.jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(2 / 4)
    assert trust.jaccard(["a"], ["z"]) == 0.0


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------
def test_fairness_detects_identity_domination():
    ranking = [
        {"feature": "driver_ham", "mean_abs_shap": 0.80},
        {"feature": "tyre_life", "mean_abs_shap": 0.10},
        {"feature": "race_progress", "mean_abs_shap": 0.10},
    ]
    a = fairness.assess(ranking, ["driver_ham"])
    assert a["identity_attribution_share"] == pytest.approx(0.80)
    assert a["expected_share_if_uniform"] == pytest.approx(1 / 3, abs=1e-4)  # assess() rounds to 4dp
    assert a["concentration_ratio"] > 2.0
    assert "predicting *who* rather than *what*" in a["reading"]


def test_fairness_reports_healthy_model_as_healthy():
    ranking = [
        {"feature": "tyre_life", "mean_abs_shap": 0.60},
        {"feature": "race_progress", "mean_abs_shap": 0.35},
        {"feature": "driver_ham", "mean_abs_shap": 0.05},
    ]
    a = fairness.assess(ranking, ["driver_ham"])
    assert a["concentration_ratio"] < 1.0
    assert "desired outcome" in a["reading"]


def test_fairness_handles_no_identity_features():
    ranking = [{"feature": "tyre_life", "mean_abs_shap": 1.0}]
    a = fairness.assess(ranking, [])
    assert a["n_identity_features"] == 0
    assert "does not arise here" in a["reading"]


def test_fairness_shares_sum_to_one():
    ranking = [
        {"feature": "driver_ham", "mean_abs_shap": 0.3},
        {"feature": "tyre_life", "mean_abs_shap": 0.7},
    ]
    a = fairness.assess(ranking, ["driver_ham"])
    assert a["identity_attribution_share"] + a["race_state_attribution_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Narrative — race-engineer language, and no clinical vocabulary
# --------------------------------------------------------------------------
def test_pit_sentence_names_the_recommendation_and_the_factors():
    s = narrative.pit_decision_sentence(
        0.82,
        [{"feature": "tyre_life", "shap_value": 0.4, "direction": "increases"},
         {"feature": "race_progress", "shap_value": -0.1, "direction": "decreases"}],
        {"tyre_life": 18.0, "race_progress": 0.6},
        "HIGH",
    )
    assert "Recommend PITTING" in s
    # the first clause is sentence-cased, so compare case-insensitively
    assert "tyre age" in s.lower() and "18" in s
    assert "HIGH" in s


def test_stay_out_sentence_when_probability_is_low():
    s = narrative.pit_decision_sentence(
        0.12, [{"feature": "tyre_life", "shap_value": -0.4, "direction": "decreases"}],
        {"tyre_life": 3.0}, "MODERATE")
    assert "Recommend STAYING OUT" in s


def test_laptime_sentence_reports_seconds():
    s = narrative.laptime_sentence(
        91.234, [{"feature": "tyre_life", "shap_value": 0.2, "direction": "increases"}],
        {"tyre_life": 20.0}, "HIGH", base_value=91.0)
    assert "91.234s" in s and "slower" in s


def test_identity_features_are_humanised_as_identity():
    assert "driver identity" in narrative.humanise("driver_ham")
    assert "team identity" in narrative.humanise("team_red_bull_racing")


def test_unknown_feature_is_not_silently_dropped():
    assert narrative.humanise("brand_new_feature_v2") == "brand new feature v2"


def test_no_clinical_vocabulary_leaks_into_explanations():
    """The reference spec is a clinical decision-support system; none of its
    vocabulary belongs in this project's output."""
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
@pytest.mark.skipif(not (OUTPUTS / "metadata" / "xai_results.json").exists(),
                    reason="run practical07.py first")
def test_committed_results_have_real_explanations_for_every_target():
    data = json.loads((OUTPUTS / "metadata" / "xai_results.json").read_text())
    for target, r in data.items():
        assert r["examples"], f"{target} has no explained rows"
        assert r["shap"]["dnn"]["ranking"], f"{target} has no SHAP ranking"
        for label, ex in r["examples"].items():
            assert ex["shap_dnn"], f"{target}/{label} has no SHAP attributions"
            assert ex["narrative"], f"{target}/{label} has no strategic explanation"
            assert 0.0 <= ex["trust"]["trust_score"] <= 1.0


@pytest.mark.skipif(not (OUTPUTS / "metadata" / "xai_results.json").exists(),
                    reason="run practical07.py first")
def test_representative_rows_are_distinct():
    data = json.loads((OUTPUTS / "metadata" / "xai_results.json").read_text())
    for target, r in data.items():
        idx = [ex["row_index"] for ex in r["examples"].values()]
        assert len(idx) == len(set(idx)), f"{target} explained the same row twice"


@pytest.mark.skipif(not (OUTPUTS / "reports").exists(), reason="run practical07.py first")
def test_every_deliverable_report_exists_and_is_non_trivial():
    expected = [
        "shap_report.md", "lime_report.md", "counterfactual_report.md",
        "trust_score_report.md", "fairness_report.md", "explainability_dashboard.md",
    ]
    for name in expected:
        p = OUTPUTS / "reports" / name
        assert p.exists(), f"missing deliverable: {name}"
        assert len(p.read_text()) > 500, f"{name} is suspiciously short"


@pytest.mark.skipif(not (OUTPUTS / "metadata" / "xai_results.json").exists(),
                    reason="run practical07.py first")
def test_reported_trust_matches_a_recomputation_from_its_own_inputs():
    """The strongest no-fabrication check available here: recompute each trust
    score from the inputs the artifact itself records."""
    data = json.loads((OUTPUTS / "metadata" / "xai_results.json").read_text())
    for target, r in data.items():
        for label, ex in r["examples"].items():
            t = ex["trust"]
            again = trust.compute(
                task=r["task"],
                dnn_prediction=t["inputs"]["dnn_prediction"],
                classical_prediction=t["inputs"]["classical_prediction"],
                shap_top=t["inputs"]["shap_top3"],
                lime_top=t["inputs"]["lime_top3"],
                target_std=t["inputs"]["target_std"],
            )
            assert again["trust_score"] == pytest.approx(t["trust_score"]), f"{target}/{label}"
