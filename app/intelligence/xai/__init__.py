"""
app.intelligence.xai
====================

Task 8 — Explainable AI.

Interprets the predictions of Task 6's persisted classical pipelines and
Task 7's deep networks so that a race engineer, not a data scientist, can see
why a strategy recommendation was made and decide how much to trust it.

Every function here is importable and callable both by the build script
(``scripts/build_all.py``) and by the API layer (``app.api.routers.xai``), so
an explanation shown in the dashboard is produced by exactly the same code
that produced the committed report.
"""
from __future__ import annotations

RANDOM_STATE = 42

__all__ = [
    "loading", "importance", "shap_analysis", "lime_analysis",
    "counterfactual", "trust", "fairness", "narrative", "visualize",
    "reports", "pipeline",
]
