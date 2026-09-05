"""
f1dl - Task 7 (Deep Learning Model Development) for F1 race strategy.

Trains Keras deep neural networks to predict expected lap time (regression)
and pit-decision probability (binary classification) from the Task 5
engineered feature matrix, and compares them honestly against classical
baselines trained on the identical folds.
"""
from __future__ import annotations

__all__ = [
    "contract",
    "splits",
    "evaluation",
    "models",
    "tuning",
    "threshold",
    "training",
    "baselines",
    "persistence",
    "visualize",
    "reports",
]

RANDOM_STATE = 42
