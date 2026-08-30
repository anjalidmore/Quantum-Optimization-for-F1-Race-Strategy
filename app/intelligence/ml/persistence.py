"""
app.intelligence.ml.persistence
==================================

Save/load complete, fitted scikit-learn pipelines (preprocessing + model
together — never just the bare estimator, since a bare estimator cannot
reproduce a prediction without the exact fitted scaler/imputer it was
trained with).
"""
from __future__ import annotations

from pathlib import Path

import joblib


def save_pipeline(pipeline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No persisted model pipeline at {path}")
    return joblib.load(path)
