"""
app.intelligence.ml.regression
=================================

Lap-time regression: predict ``target_laptime`` from information available
before the lap starts.

Model definitions, bounded hyperparameter grids, and feature-importance
extraction for the five regression models required by Task 6. XGBoost is
optional: if the package cannot be imported (missing, or — as observed in
this environment before ``libomp`` was installed — a native library load
failure) it is skipped with an explicit, honest status rather than a
fabricated result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from app.intelligence.features.contract import FeatureContract
from app.intelligence.ml.preprocessing import build_model_pipeline

try:
    from xgboost import XGBRegressor

    _XGBOOST_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only when xgboost/libomp missing
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def xgboost_available() -> bool:
    return XGBRegressor is not None


def xgboost_unavailable_reason() -> str | None:
    return _XGBOOST_IMPORT_ERROR


@dataclass(frozen=True)
class RegressionModelSpec:
    name: str
    needs_scaling: bool
    build_estimator: Callable[[int], object]
    param_grid: dict[str, list]
    supports_importance: str  # "coefficients" | "tree" | "none"


RANDOM_STATE = 42

REGRESSION_MODEL_SPECS: list[RegressionModelSpec] = [
    RegressionModelSpec(
        name="linear_regression",
        needs_scaling=True,
        build_estimator=lambda rs: LinearRegression(),
        param_grid={},
        supports_importance="coefficients",
    ),
    RegressionModelSpec(
        name="decision_tree",
        needs_scaling=False,
        build_estimator=lambda rs: DecisionTreeRegressor(random_state=rs),
        param_grid={"max_depth": [3, 5, 8], "min_samples_leaf": [1, 5, 10]},
        supports_importance="tree",
    ),
    RegressionModelSpec(
        name="random_forest",
        needs_scaling=False,
        build_estimator=lambda rs: RandomForestRegressor(random_state=rs, n_jobs=-1),
        param_grid={"n_estimators": [100, 300], "max_depth": [5, 10, None]},
        supports_importance="tree",
    ),
    RegressionModelSpec(
        name="svr",
        needs_scaling=True,
        build_estimator=lambda rs: SVR(),
        param_grid={"C": [1.0, 10.0], "gamma": ["scale", "auto"], "epsilon": [0.05, 0.1]},
        supports_importance="none",
    ),
]

if xgboost_available():
    REGRESSION_MODEL_SPECS.append(
        RegressionModelSpec(
            name="xgboost",
            needs_scaling=False,
            build_estimator=lambda rs: XGBRegressor(
                random_state=rs, objective="reg:squarederror", n_jobs=-1, verbosity=0
            ),
            param_grid={
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            supports_importance="tree",
        )
    )


def build_pipeline_factory(spec: RegressionModelSpec, feature_list: list[str], contract: FeatureContract):
    def _factory():
        estimator = spec.build_estimator(RANDOM_STATE)
        return build_model_pipeline(estimator, feature_list, contract, scale=spec.needs_scaling)

    return _factory


def _transformed_feature_names(fitted_pipeline) -> list[str]:
    """The ColumnTransformer may reorder columns (numeric block, then binary
    block, ...), so feature-importance values must be zipped against its
    actual output order, not the original ``feature_list`` order."""
    raw_names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def extract_feature_importance(spec: RegressionModelSpec, fitted_pipeline, feature_list: list[str]) -> dict | None:
    model = fitted_pipeline.named_steps["model"]
    names = _transformed_feature_names(fitted_pipeline)
    if spec.supports_importance == "coefficients":
        coefs = np.asarray(model.coef_).ravel()
        return {f: float(abs(c)) for f, c in zip(names, coefs)}
    if spec.supports_importance == "tree":
        importances = np.asarray(model.feature_importances_).ravel()
        return {f: float(i) for f, i in zip(names, importances)}
    return None
