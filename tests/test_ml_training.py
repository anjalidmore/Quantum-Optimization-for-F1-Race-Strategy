"""Integration tests for the real Task 6 training pipeline.

These run the actual pipeline once (module-scoped fixture) and assert on its
real outputs — no mocked models, no hand-typed metric values.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.paths import ML_MODELS_LAPTIME_DIR, ML_MODELS_PIT_DIR
from app.intelligence.ml import classification as clf_mod
from app.intelligence.ml import regression as reg_mod
from app.intelligence.ml.persistence import load_pipeline
from app.intelligence.ml.pipeline import train_all


@pytest.fixture(scope="module")
def training_summary():
    return train_all()


def test_pipeline_selects_a_best_model_for_each_task(training_summary):
    assert training_summary["best_regression_model"] is not None
    assert training_summary["best_classification_model"] is not None


def test_all_configured_regression_models_trained_or_honestly_skipped(training_summary):
    trained_names = {r["model"] for r in training_summary["regression_comparison"] if r.get("status") != "skipped"}
    expected = {spec.name for spec in reg_mod.REGRESSION_MODEL_SPECS}
    assert expected <= trained_names

    if not reg_mod.xgboost_available():
        skipped = {r["model"] for r in training_summary["regression_comparison"] if r.get("status") == "skipped"}
        assert "xgboost" in skipped


def test_all_configured_classification_models_trained_or_honestly_skipped(training_summary):
    trained_names = {r["model"] for r in training_summary["classification_comparison"] if r.get("status") != "skipped"}
    expected = {spec.name for spec in clf_mod.CLASSIFICATION_MODEL_SPECS}
    assert expected <= trained_names


def test_regression_cv_and_test_metrics_are_finite(training_summary):
    for row in training_summary["regression_comparison"]:
        if row.get("status") == "skipped":
            continue
        for key in ("cv_mae", "cv_rmse", "test_mae", "test_rmse"):
            value = row[key]
            assert value is not None
            assert math.isfinite(value)


def test_classification_cv_metrics_are_finite_where_defined(training_summary):
    for row in training_summary["classification_comparison"]:
        if row.get("status") == "skipped":
            continue
        for key in ("cv_roc_auc", "cv_pr_auc", "cv_f1"):
            value = row[key]
            if value is not None:
                assert math.isfinite(value)


def test_saved_regression_pipelines_load_and_predict_with_correct_shape():
    for spec in reg_mod.REGRESSION_MODEL_SPECS:
        path = ML_MODELS_LAPTIME_DIR / f"{spec.name}.joblib"
        assert path.exists(), f"missing persisted artifact for {spec.name}"
        pipeline = load_pipeline(path)
        # Build a tiny synthetic frame matching the pipeline's expected columns.
        n_features = pipeline.named_steps["preprocess"].feature_names_in_
        import pandas as pd

        X = pd.DataFrame(np.zeros((3, len(n_features))), columns=n_features)
        preds = pipeline.predict(X)
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))


def test_saved_pipeline_is_deterministic_same_input_same_prediction():
    path = ML_MODELS_LAPTIME_DIR / "linear_regression.joblib"
    pipeline_a = load_pipeline(path)
    pipeline_b = load_pipeline(path)

    import pandas as pd

    cols = pipeline_a.named_steps["preprocess"].feature_names_in_
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(5, len(cols))), columns=cols)

    preds_a = pipeline_a.predict(X)
    preds_b = pipeline_b.predict(X)
    np.testing.assert_array_equal(preds_a, preds_b)


def test_saved_classification_pipelines_load_and_predict_proba():
    for spec in clf_mod.CLASSIFICATION_MODEL_SPECS:
        path = ML_MODELS_PIT_DIR / f"{spec.name}.joblib"
        assert path.exists(), f"missing persisted artifact for {spec.name}"
        pipeline = load_pipeline(path)
        cols = pipeline.named_steps["preprocess"].feature_names_in_
        import pandas as pd

        X = pd.DataFrame(np.zeros((3, len(cols))), columns=cols)
        preds = pipeline.predict(X)
        assert preds.shape == (3,)
        assert set(np.unique(preds)) <= {0, 1}
        proba = pipeline.predict_proba(X)
        assert proba.shape == (3, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
