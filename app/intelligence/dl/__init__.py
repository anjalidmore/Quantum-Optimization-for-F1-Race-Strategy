"""
app.intelligence.dl
===================

Task 7 — Deep Learning Model Development.

Keras deep neural networks predicting expected lap time (regression) and
pit-decision probability (binary classification) from the Task 5 feature
contract.

This package deliberately does **not** re-implement data loading, splitting or
metric computation. It imports ``app.intelligence.ml``'s ``data_contract``,
``splits`` and ``evaluation`` directly, so Task 7's numbers and Task 6's are
produced by the same code and any difference between them is attributable to
the model rather than to the harness.
"""
from __future__ import annotations

RANDOM_STATE = 42

__all__ = ["models", "training", "tuning", "persistence", "visualize", "pipeline"]
