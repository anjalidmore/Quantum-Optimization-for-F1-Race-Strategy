"""
f1dl.baselines
==============

Classical baselines, trained on the identical folds and the identical
untouched test set as the deep networks.

Task 7's stated purpose is to establish *whether* a DNN improves on Task 6's
classical models - which requires a like-for-like comparison, not a
comparison against numbers copied from another branch's report. On
``task-mode`` each practical is self-contained, and Task 6 lives in the
unified app on ``proj-mode``, so the baselines are re-trained here using the
same estimator specifications and the same leakage-safe fold scaling.

(On ``proj-mode``, Phase 3 wires the comparison to Task 6's real persisted
``.joblib`` pipelines and its ``model_registry.json`` instead.)
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import evaluation, training

RANDOM_STATE = 42


def regression_baselines() -> dict:
    """Same estimator family and hyperparameters Task 6 uses for lap time."""
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5),
        "random_forest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_estimators=200, max_depth=8, n_jobs=-1
        ),
        # A mean predictor makes R2 interpretable: R2 <= 0 means the model is
        # no better than this, which is the honest reference point for the
        # negative test R2 Task 6 already reports.
        "mean_baseline": DummyRegressor(strategy="mean"),
    }


def classification_baselines() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=5, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=200, max_depth=8,
            class_weight="balanced", n_jobs=-1,
        ),
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
    }


def fit_and_score(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    numeric_mask: np.ndarray,
    *,
    task: str,
) -> dict:
    """Fit on train, score on test, using the same fold-fitted scaling the
    networks get so neither side has a preprocessing advantage."""
    Xt, Xs, _ = training.scale_fit_transform(X_train, X_test, numeric_mask)
    estimator.fit(Xt, y_train)

    if task == "regression":
        return evaluation.regression_metrics(y_test, estimator.predict(Xs))

    proba = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(Xs)[:, 1]
    return evaluation.classification_metrics(y_test, estimator.predict(Xs), y_proba=proba)
