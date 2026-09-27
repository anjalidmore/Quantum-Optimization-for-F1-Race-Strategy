"""
Decision-threshold tuning, and the model-selection rules that depend on it.

These cover the Phase 2 model-quality fixes: a classifier that ranks well but
never fires, and a regressor selected on CV that loses to the mean out of
sample.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.intelligence.ml.selection import (
    ModelResult,
    rank_classification_models,
    rank_regression_models,
)
from app.intelligence.ml.threshold import (
    DEFAULT_THRESHOLD,
    apply,
    metrics_at,
    tune_threshold,
)


# ---------------------------------------------------------------------------
# metrics_at
# ---------------------------------------------------------------------------
def test_metrics_at_matches_a_hand_computation():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    m = metrics_at(y, p, threshold=0.5)
    # predicted positive: 0.9 (tp) and 0.6 (fp); 0.4 is a fn
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_expected_cost_weights_false_negatives_more_heavily():
    y = np.array([1, 0])
    one_fn = metrics_at(y, np.array([0.1, 0.1]), 0.5, fn_cost=5.0, fp_cost=1.0)
    one_fp = metrics_at(y, np.array([0.9, 0.9]), 0.5, fn_cost=5.0, fp_cost=1.0)
    assert one_fn["expected_cost"] == 5.0
    assert one_fp["expected_cost"] == 1.0


def test_metrics_are_zero_not_nan_when_nothing_is_predicted_positive():
    m = metrics_at(np.array([1, 0, 0]), np.array([0.1, 0.1, 0.1]), 0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


# ---------------------------------------------------------------------------
# tune_threshold
# ---------------------------------------------------------------------------
def test_tuning_rescues_a_classifier_that_never_fires_at_0_5():
    """The exact failure this module exists for: a model that ranks perfectly
    but whose probabilities all sit below 0.5, so `predict()` says 'no' to
    everything and F1 is 0."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 380)
    # Perfect ranking, but compressed into [0, 0.3] — nothing crosses 0.5.
    p = np.concatenate([rng.uniform(0.20, 0.30, 20), rng.uniform(0.00, 0.15, 380)])

    baseline = metrics_at(y, p, DEFAULT_THRESHOLD)
    assert baseline["f1"] == 0.0, "fixture should start degenerate"

    choice = tune_threshold(y, p, objective="f1")
    assert choice.threshold < DEFAULT_THRESHOLD
    assert choice.f1 > 0.9, f"a perfectly-ranked signal should be recoverable, got {choice.f1}"
    assert choice.baseline["f1"] == 0.0
    assert "0.5" in choice.note


def test_tuning_never_makes_f1_worse_than_the_default():
    """0.5 is inside the candidate range, so the optimum can only tie or beat
    it on the data it was tuned on."""
    rng = np.random.default_rng(1)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = (rng.random(300) < 0.1).astype(int)
        p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 300), 0, 1)
        choice = tune_threshold(y, p, objective="f1")
        assert choice.f1 >= choice.baseline["f1"] - 1e-9


def test_expected_cost_objective_prefers_recall_when_misses_are_costly():
    rng = np.random.default_rng(2)
    y = (rng.random(400) < 0.08).astype(int)
    p = np.clip(y * 0.35 + rng.normal(0.3, 0.15, 400), 0, 1)
    f1_choice = tune_threshold(y, p, objective="f1")
    cost_choice = tune_threshold(y, p, objective="expected_cost", fn_cost=20.0, fp_cost=1.0)
    # Making misses 20x as expensive should not push the threshold upward.
    assert cost_choice.threshold <= f1_choice.threshold + 1e-9
    assert cost_choice.recall >= f1_choice.recall - 1e-9


def test_no_positives_keeps_the_default_and_says_why():
    """A cut-off cannot be learned from data with none of the class it detects.
    Returning a number anyway would be a fabricated result."""
    choice = tune_threshold(np.zeros(50, dtype=int), np.random.default_rng(0).random(50))
    assert choice.threshold == DEFAULT_THRESHOLD
    assert choice.n_positive == 0
    assert "no threshold is learnable" in choice.note


def test_all_positives_also_keeps_the_default():
    choice = tune_threshold(np.ones(50, dtype=int), np.random.default_rng(0).random(50))
    assert choice.threshold == DEFAULT_THRESHOLD
    assert "no threshold is learnable" in choice.note


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="Unknown objective"):
        tune_threshold(np.array([0, 1]), np.array([0.2, 0.8]), objective="accuracy")


def test_metadata_records_both_the_chosen_and_default_operating_points():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.1).astype(int)
    p = np.clip(y * 0.4 + rng.normal(0.25, 0.15, 300), 0, 1)
    meta = tune_threshold(y, p).to_metadata()
    assert set(meta) >= {"threshold", "objective", "chosen_on", "at_threshold", "at_default_0.5"}
    assert meta["chosen_on"] == "pooled out-of-fold CV predictions"
    assert set(meta["at_threshold"]) == {"f1", "precision", "recall", "expected_cost"}


def test_apply_is_inclusive_at_the_threshold():
    assert list(apply(np.array([0.49, 0.50, 0.51]), 0.50)) == [0, 1, 1]


# ---------------------------------------------------------------------------
# Selection: the generalisation guard
# ---------------------------------------------------------------------------
def _reg(name, cv_mae, test_r2):
    return ModelResult(
        model_name=name, task="laptime_regression",
        cv_summary={"mae": {"mean": cv_mae}, "rmse": {"mean": 1.0}, "r2": {"mean": 0.3}},
        test_metrics={"mae": 1.0, "rmse": 1.0, "r2": test_r2},
        fit_seconds=0.1, best_params={},
    )


def test_guard_overrides_a_cv_winner_that_loses_to_the_mean():
    """The exact Task 6 failure: decision_tree won on CV MAE with test R2 -0.17
    while svr generalised."""
    rows = rank_regression_models([
        _reg("cv_winner_but_broken", cv_mae=1.19, test_r2=-0.17),
        _reg("generalises", cv_mae=1.38, test_r2=0.30),
    ])
    selected = next(r for r in rows if r["selected"])
    assert selected["model"] == "generalises"
    assert "generalisation guard" in selected["selection_warning"]
    assert "cv_winner_but_broken" in selected["selection_warning"]


def test_guard_does_not_fire_when_the_cv_winner_generalises():
    rows = rank_regression_models([
        _reg("good", cv_mae=1.0, test_r2=0.5),
        _reg("worse", cv_mae=1.5, test_r2=0.4),
    ])
    selected = next(r for r in rows if r["selected"])
    assert selected["model"] == "good"
    assert "selection_warning" not in selected


def test_guard_picks_the_best_cv_performer_among_those_that_generalise():
    """The override must be minimal — it changes *which* model, not the metric."""
    rows = rank_regression_models([
        _reg("broken", cv_mae=1.0, test_r2=-0.5),
        _reg("ok_better_cv", cv_mae=1.2, test_r2=0.1),
        _reg("ok_worse_cv", cv_mae=1.4, test_r2=0.9),
    ])
    selected = next(r for r in rows if r["selected"])
    assert selected["model"] == "ok_better_cv", "guard should not switch to maximising test R2"


def test_guard_reports_honestly_when_nothing_generalises():
    rows = rank_regression_models([
        _reg("a", cv_mae=1.0, test_r2=-0.5),
        _reg("b", cv_mae=1.2, test_r2=-0.1),
    ])
    selected = next(r for r in rows if r["selected"])
    assert selected["model"] == "a"
    assert "no model in this run generalises" in selected["selection_warning"]


# ---------------------------------------------------------------------------
# Selection: PR-AUC as the classification primary
# ---------------------------------------------------------------------------
def _clf(name, pr_auc, roc_auc):
    return ModelResult(
        model_name=name, task="pit_decision_classification",
        cv_summary={"pr_auc": {"mean": pr_auc}, "roc_auc": {"mean": roc_auc}, "f1": {"mean": 0.2}},
        test_metrics={"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": 0.2},
        fit_seconds=0.1, best_params={},
        threshold={"threshold": 0.3, "objective": "f1"},
    )


def test_classification_selects_on_pr_auc_not_roc_auc():
    """At 4.8% prevalence ROC-AUC stays high for a model that never fires, so
    it must not decide the winner."""
    rows = rank_classification_models([
        _clf("high_roc_low_pr", pr_auc=0.10, roc_auc=0.98),
        _clf("lower_roc_high_pr", pr_auc=0.40, roc_auc=0.85),
    ])
    selected = next(r for r in rows if r["selected"])
    assert selected["model"] == "lower_roc_high_pr"


def test_classification_rows_expose_the_threshold_and_operating_point():
    rows = rank_classification_models([_clf("m", pr_auc=0.4, roc_auc=0.9)])
    row = rows[0]
    for key in ("decision_threshold", "threshold_objective", "test_precision", "test_recall", "test_f1"):
        assert key in row, f"{key} missing — the dashboard needs it to show the operating point"
    assert row["decision_threshold"] == 0.3
