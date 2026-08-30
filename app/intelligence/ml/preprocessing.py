"""
app.intelligence.ml.preprocessing
====================================

Model-specific preprocessing pipelines.

The Task 5 contract (``feature_metadata.json``) already tells us which
selected features are continuous and require scaling versus which are
already binary indicators:

    numeric_features_requiring_scaling
    binary_features_no_scaling_needed

and explicitly states scaling must be *fit inside each CV fold* to avoid
leakage: "Fit the scaler inside each CV fold." We satisfy that not by
scaling the data up front, but by putting the scaler inside an
``sklearn.pipeline.Pipeline`` alongside the estimator — a fresh scaler
instance is fit on the training fold only, every time the pipeline is
cloned and re-fit by the CV loop.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.intelligence.features.contract import FeatureContract


def build_preprocessor(
    feature_list: list[str], contract: FeatureContract, scale: bool
) -> ColumnTransformer:
    """A ColumnTransformer that imputes (defensive; the contract's data has
    no missing values) and, for models that need it, standard-scales the
    continuous features. Binary indicator features are passed through
    unscaled either way — scaling a 0/1 indicator changes nothing but its
    interpretability.
    """
    numeric_cols = [c for c in feature_list if c in contract.numeric_features_requiring_scaling]
    binary_cols = [c for c in feature_list if c in contract.binary_features_no_scaling_needed]
    other_cols = [c for c in feature_list if c not in numeric_cols and c not in binary_cols]

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if binary_cols:
        transformers.append(
            ("binary", SimpleImputer(strategy="most_frequent"), binary_cols)
        )
    if other_cols:
        # Features not classified either way by the Task 5 contract still
        # get scaled alongside the numeric ones when the model needs it, so
        # nothing silently escapes preprocessing.
        transformers.append(("other", numeric_pipeline, other_cols))

    return ColumnTransformer(transformers, remainder="drop")


def build_model_pipeline(estimator, feature_list: list[str], contract: FeatureContract, scale: bool) -> Pipeline:
    preprocessor = build_preprocessor(feature_list, contract, scale=scale)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
