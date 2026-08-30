"""
app.intelligence.features
==========================

Task 5 — Feature Engineering & Feature Selection.

This module does not recompute features. Feature engineering and selection
were performed in ``docs/notebooks/task5_feature_engineering.ipynb`` against
``data/processed/fastf1_laps_clean.csv`` (Task 4's output) and the results
were frozen as the Task 6 data contract:

    data/processed/f1_features_selected.csv
    data/processed/feature_metadata.json

:mod:`app.intelligence.features.contract` is the single place that reads
those two files. Every downstream consumer (Task 6 ML, the API, the
frontend) goes through it instead of re-parsing the CSV/JSON directly, so
the selected feature lists are never hard-coded in more than one place.
"""

from app.intelligence.features.contract import FeatureContract, load_feature_contract

__all__ = ["FeatureContract", "load_feature_contract"]
