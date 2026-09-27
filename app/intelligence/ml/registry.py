"""
app.intelligence.ml.registry
===============================

A machine-readable registry of every trained Task 6 model — the file the
backend API and, later, Task 8 (Explainable AI) read instead of re-deriving
"what models exist and where are they" from the filesystem.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import sklearn

from app.core.paths import ML_MODEL_REGISTRY_JSON


@dataclass
class ModelRegistryEntry:
    model_name: str
    task: str  # == target ("target_laptime" | "target_pit_next_lap"); kept as a
    # separate field so a future task grouping (e.g. adding DL models under the
    # same target in Task 7) doesn't have to repurpose this one.
    target: str
    features: list[str]
    validation: dict
    metrics: dict
    artifact: str
    hyperparameters: dict
    random_state: int
    dataset: str
    training_rows: int
    test_rows: int
    cv_folds: int
    is_selected_best: bool
    synthetic_data_warning: bool
    xgboost_status: str | None = None
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    software_versions: dict = field(
        default_factory=lambda: {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        }
    )


def write_registry(entries: list[ModelRegistryEntry], path: Path = ML_MODEL_REGISTRY_JSON) -> Path:
    """Write Task 6's entries, **preserving** any other family's.

    Task 7 extends this same file rather than keeping a parallel registry, so
    there is one source of truth for "what models exist". That only works if
    retraining Task 6 does not silently delete Task 7's rows — which a naive
    whole-file overwrite would do, leaving the API advertising deep models
    that are no longer registered. Entries carrying a ``family`` other than
    "classical" are therefore carried across unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    preserved: list[dict] = []
    existing = load_registry(path)
    if existing:
        preserved = [
            m for m in existing.get("models", [])
            if m.get("family") not in (None, "classical")
        ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "Task 6 - Machine Learning Model Development",
        "models": [asdict(e) for e in entries] + preserved,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_registry(path: Path = ML_MODEL_REGISTRY_JSON) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
