"""
app.intelligence.dl.persistence
===============================

Saving and reloading the Task 7 networks.

Layout, one folder per target::

    artifacts/models/deep_learning/<laptime|pit_decision>/
        f1_dnn_model.h5        the trained network (HDF5)
        feature_scaler.joblib  StandardScaler fitted on the training rows only
        target_scaler.joblib   lap-time target scaler (regression only)
        model_spec.json        feature order, which columns are scaled, format

These live inside the private ``models/`` tree next to Task 6's pipelines, so
trained weights are never exposed by the API's static artifact mount.

**HDF5 (.h5).** The deliverable format. Before saving, the model is recompiled
with a standard loss: the classifier trains with a custom class-weighted loss
(see ``models.weighted_binary_crossentropy``), and a file that needs custom code
to deserialise is a file other people cannot open. Recompiling changes nothing
about the trained weights. ``save`` then reloads the file and refuses to
continue unless it reproduces the in-memory model's predictions exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

from app.core.paths import TARGET_DIRNAME  # noqa: E402

MODEL_FILENAME = "f1_dnn_model.h5"
MODEL_EXTENSION = ".h5"


@dataclass(frozen=True)
class SavedModel:
    model_path: Path
    scaler_path: Path
    spec_path: Path
    size_bytes: int
    reload_verified: bool


def target_dir(root: Path, target: str) -> Path:
    """``root`` is ``artifacts/models/deep_learning`` (or a redirected copy)."""
    return Path(root) / TARGET_DIRNAME[target]


def save(model, scaler, features: list[str], numeric_mask: np.ndarray, target: str,
         models_root: Path, y_scaler=None, X_check: np.ndarray | None = None) -> SavedModel:
    out_dir = target_dir(models_root, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME
    scaler_path = out_dir / "feature_scaler.joblib"
    y_scaler_path = out_dir / "target_scaler.joblib"
    spec_path = out_dir / "model_spec.json"

    task_loss = "mse" if target == "target_laptime" else "binary_crossentropy"
    model.compile(optimizer=model.optimizer, loss=task_loss,
                  metrics=["mae"] if target == "target_laptime" else ["accuracy"])
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    if y_scaler is not None:
        joblib.dump(y_scaler, y_scaler_path)
    elif y_scaler_path.exists():
        y_scaler_path.unlink()

    verified = False
    if X_check is not None:
        reloaded = keras.saving.load_model(model_path, compile=False)
        if not np.allclose(reloaded.predict(X_check, verbose=0), model.predict(X_check, verbose=0), atol=1e-6):
            raise RuntimeError(f"{model_path} does not reproduce the trained model's predictions.")
        verified = True

    spec_path.write_text(json.dumps({
        "target": target,
        "features": features,
        "numeric_mask": [bool(b) for b in numeric_mask],
        "model_file": MODEL_FILENAME,
        "feature_scaler_file": scaler_path.name,
        "target_scaler_file": y_scaler_path.name if y_scaler is not None else None,
        "format": "HDF5 (.h5), Keras 3",
        "load_with": f"keras.saving.load_model('{MODEL_FILENAME}', compile=False)",
        "reload_verified": verified,
    }, indent=2))
    return SavedModel(model_path, scaler_path, spec_path, model_path.stat().st_size, verified)


def load(target: str, models_root: Path):
    """Reload a saved network, its scalers and its feature spec."""
    d = target_dir(models_root, target)
    spec = json.loads((d / "model_spec.json").read_text())
    model = keras.saving.load_model(d / spec["model_file"], compile=False)
    scaler = joblib.load(d / spec["feature_scaler_file"])
    y_scaler = joblib.load(d / spec["target_scaler_file"]) if spec.get("target_scaler_file") else None
    return model, scaler, y_scaler, spec
