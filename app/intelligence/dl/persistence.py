"""
app.intelligence.dl.persistence
===============================

Saving and reloading trained networks.

Format note (a deliberate, documented divergence from the reference spec):
the CDSS task table names ``.h5`` as the deliverable. Under Keras 3 the HDF5
path is legacy - a model saves to ``.h5`` but fails to load back
(``Could not deserialize 'keras.metrics.mse'``), verified on this
installation. The native ``.keras`` archive round-trips correctly, so that is
what is shipped. A model that cannot be reloaded is not a deliverable.

Each model is saved alongside its fitted ``StandardScaler`` and the feature
order it expects, because a network's weights are meaningless without the
exact transform and column order used at training time.
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

MODEL_EXTENSION = ".keras"
SPEC_EXTENSION_IN_REFERENCE = ".h5"


@dataclass(frozen=True)
class SavedModel:
    model_path: Path
    scaler_path: Path
    spec_path: Path


def save(model, scaler, features: list[str], numeric_mask: np.ndarray, target: str,
         out_dir: Path, y_scaler=None) -> SavedModel:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{target}{MODEL_EXTENSION}"
    scaler_path = out_dir / f"{target}_scaler.joblib"
    spec_path = out_dir / f"{target}_spec.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    y_scaler_path = out_dir / f"{target}_target_scaler.joblib"
    if y_scaler is not None:
        joblib.dump(y_scaler, y_scaler_path)
    spec_path.write_text(
        json.dumps(
            {
                "target": target,
                "features": features,
                "numeric_mask": [bool(b) for b in numeric_mask],
                "model_file": model_path.name,
                "scaler_file": scaler_path.name,
                "target_scaler_file": y_scaler_path.name if y_scaler is not None else None,
                "format": MODEL_EXTENSION,
                "format_note": (
                    f"Reference spec names {SPEC_EXTENSION_IN_REFERENCE}; Keras 3 cannot "
                    f"reload HDF5 models saved this way, so the native {MODEL_EXTENSION} "
                    f"archive is used instead."
                ),
            },
            indent=2,
        )
    )
    return SavedModel(model_path=model_path, scaler_path=scaler_path, spec_path=spec_path)


def load(target: str, out_dir: Path):
    """Reload a saved model, its scaler and its feature spec."""
    spec = json.loads((out_dir / f"{target}_spec.json").read_text())
    model = keras.saving.load_model(out_dir / spec["model_file"])
    scaler = joblib.load(out_dir / spec["scaler_file"])
    y_scaler = None
    if spec.get("target_scaler_file"):
        y_scaler = joblib.load(out_dir / spec["target_scaler_file"])
    return model, scaler, y_scaler, spec
