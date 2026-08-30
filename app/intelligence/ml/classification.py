"""
app.intelligence.ml.classification
=====================================

Pit-decision classification: predict ``target_pit_next_lap`` — should the
driver pit at the end of this lap?

Pit events are rare (~3% of rows) and, in this synthetic session, clustered
almost deterministically at specific laps (see ``feature_metadata.json``'s
``validation_scores.caveat``). Every model here uses class weighting /
imbalance handling so a trivial "always predict no-pit" classifier is not
rewarded by accuracy alone — which is also why Task 6 ranks classifiers by
ROC-AUC / PR-AUC / F1, not accuracy.

XGBoost is optional, mirroring ``regression.py``: skipped with an honest
status if the package or its native runtime is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.intelligence.features.contract import FeatureContract
from app.intelligence.ml.preprocessing import build_model_pipeline

try:
    from xgboost import XGBClassifier

    _XGBOOST_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only when xgboost/libomp missing
    XGBClassifier = None
    _XGBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def xgboost_available() -> bool:
    return XGBClassifier is not None


def xgboost_unavailable_reason() -> str | None:
    return _XGBOOST_IMPORT_ERROR


@dataclass(frozen=True)
class ClassificationModelSpec:
    name: str
    needs_scaling: bool
    build_estimator: Callable[[int], object]
    param_grid: dict[str, list]
    supports_importance: str  # "coefficients" | "tree" | "none"


RANDOM_STATE = 42

CLASSIFICATION_MODEL_SPECS: list[ClassificationModelSpec] = [
    ClassificationModelSpec(
        name="logistic_regression",
        needs_scaling=True,
        build_estimator=lambda rs: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=rs
        ),
        param_grid={"C": [0.1, 1.0, 10.0]},
        supports_importance="coefficients",
    ),
    ClassificationModelSpec(
        name="decision_tree",
        needs_scaling=False,
        build_estimator=lambda rs: DecisionTreeClassifier(
            class_weight="balanced", random_state=rs
        ),
        param_grid={"max_depth": [3, 5, 8], "min_samples_leaf": [1, 5, 10]},
        supports_importance="tree",
    ),
    ClassificationModelSpec(
        name="random_forest",
        needs_scaling=False,
        build_estimator=lambda rs: RandomForestClassifier(
            class_weight="balanced", random_state=rs, n_jobs=-1
        ),
        param_grid={"n_estimators": [100, 300], "max_depth": [5, 10, None]},
        supports_importance="tree",
    ),
    ClassificationModelSpec(
        name="svm",
        needs_scaling=True,
        build_estimator=lambda rs: SVC(
            class_weight="balanced", probability=True, random_state=rs
        ),
        param_grid={"C": [1.0, 10.0], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
        supports_importance="none",
    ),
]

if xgboost_available():

    def _xgb_classifier(rs: int):
        return XGBClassifier(
            random_state=rs,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )

    CLASSIFICATION_MODEL_SPECS.append(
        ClassificationModelSpec(
            name="xgboost",
            needs_scaling=False,
            build_estimator=_xgb_classifier,
            param_grid={
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            supports_importance="tree",
        )
    )


def build_pipeline_factory(spec: ClassificationModelSpec, feature_list: list[str], contract: FeatureContract):
    def _factory():
        estimator = spec.build_estimator(RANDOM_STATE)
        return build_model_pipeline(estimator, feature_list, contract, scale=spec.needs_scaling)

    return _factory


def _transformed_feature_names(fitted_pipeline) -> list[str]:
    raw_names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def extract_feature_importance(
    spec: ClassificationModelSpec, fitted_pipeline, feature_list: list[str]
) -> dict | None:
    model = fitted_pipeline.named_steps["model"]
    names = _transformed_feature_names(fitted_pipeline)
    if spec.supports_importance == "coefficients":
        coefs = np.asarray(model.coef_).ravel()
        return {f: float(abs(c)) for f, c in zip(names, coefs)}
    if spec.supports_importance == "tree":
        importances = np.asarray(model.feature_importances_).ravel()
        return {f: float(i) for f, i in zip(names, importances)}
    return None
