"""
f1xai - Task 8 (Explainable AI) for F1 race strategy.

Interprets the predictions of the classical models (Task 6's family) and the
Task 7 deep networks so that a race engineer - not a data scientist - can see
*why* a strategy recommendation was made and decide how much to trust it.

Every explanation here is computed from a real fitted model on real test-set
rows. Nothing in this package templates a plausible-sounding number.
"""
from __future__ import annotations

RANDOM_STATE = 42

__all__ = [
    "loading", "importance", "shap_analysis", "lime_analysis",
    "counterfactual", "trust", "fairness", "narrative", "visualize", "reports",
]
