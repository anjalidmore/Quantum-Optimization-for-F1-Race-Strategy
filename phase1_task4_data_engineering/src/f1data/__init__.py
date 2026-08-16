"""
f1data — F1 Data Engineering & EDA (Phase 1, Task 4)
====================================================

A production data-engineering pipeline over the historical Kaggle dataset and
FastF1 laps:

* ``schemas``   — canonical column schemas matching the real data sources.
* ``synthetic`` — realistic sample-data generator (runnable without downloads).
* ``pipeline``  — load / clean / encode / scale with a full audit trail.
* ``eda``       — statistics, correlation, quality scoring, domain analyses.
* ``visualize`` — analytics figures and a composite dashboard.
* ``reports``   — Markdown deliverable generators.

No machine learning modelling and no quantum computing — this is the classical
data foundation for Phase 2.
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["schemas", "synthetic", "pipeline", "eda", "visualize", "reports"]
