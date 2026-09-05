"""
app.api.routers.dl
==================

Task 7 — Deep Learning endpoints, mirroring ``ml.py``'s shape.

Every response is read from artifacts the real Task 7 pipeline generated, or
computed by the saved network itself. Nothing here is fabricated: if Task 7
has not been run, these endpoints say so with a 404/503 rather than inventing
a number.
"""
from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DeepPredictionResponse,
    LapTimePredictionRequest,
    PitPredictionRequest,
)
from app.core.paths import (
    DL_COMPARISON_JSON,
    DL_HISTORY_JSON,
    DL_METRICS_JSON,
    DL_MODELS_DIR,
    ML_MODEL_REGISTRY_JSON,
)
from app.intelligence.dl import persistence as dl_persistence
from app.intelligence.dl import training as dl_training
from app.intelligence.features.contract import load_feature_contract

router = APIRouter(prefix="/api/dl", tags=["deep-learning"])

_TARGETS = ("target_laptime", "target_pit_next_lap")


def _read_json(path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No Task 7 artifact at {path}. Run the deep-learning stage "
                   "(python scripts/build_all.py) to generate it.",
        )
    return json.loads(path.read_text())


def _load(target: str):
    try:
        return dl_persistence.load(target, DL_MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No trained deep model for {target!r}. Run scripts/build_all.py first.",
        ) from exc


@router.get("/models")
def list_models():
    """The Task 7 entries in the shared model registry. Task 7 extends the
    existing registry rather than keeping a parallel one, so this is a filtered
    view of the same file ``/api/ml/models`` serves."""
    if not ML_MODEL_REGISTRY_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="No model registry found. Run the training pipeline (scripts/build_all.py) first.",
        )
    registry = json.loads(ML_MODEL_REGISTRY_JSON.read_text())
    deep = [m for m in registry.get("models", []) if m.get("family") == "deep"]
    if not deep:
        raise HTTPException(
            status_code=404,
            detail="The model registry contains no Task 7 deep models. Run scripts/build_all.py.",
        )
    return {"generated_at": registry.get("generated_at"), "task": "Task 7 - Deep Learning", "models": deep}


@router.get("/metrics")
def get_metrics():
    return _read_json(DL_METRICS_JSON)


@router.get("/comparison")
def get_comparison():
    """Deep network versus Task 6's classical models on the identical
    chronological holdout."""
    return _read_json(DL_COMPARISON_JSON)


@router.get("/history")
def get_history():
    """Per-epoch loss/metric curves — the primary diagnostic for whether a
    network was still learning, had plateaued, or was overfitting."""
    return _read_json(DL_HISTORY_JSON)


@router.get("/artifacts")
def get_artifacts():
    """Which Task 7 files exist on disk, so the frontend can link only to
    figures that are really there."""
    from app.core.paths import ML_FIGURES_DIR

    figures = sorted(p.name for p in ML_FIGURES_DIR.glob("dl_*.png")) if ML_FIGURES_DIR.exists() else []
    models = sorted(p.name for p in DL_MODELS_DIR.glob(f"*{dl_persistence.MODEL_EXTENSION}")) \
        if DL_MODELS_DIR.exists() else []
    return {
        "figures": [f"figures/{n}" for n in figures],
        "models": [f"models/dl/{n}" for n in models],
        "reports": ["reports/dl_evaluation_report.md", "reports/dl_hyperparameter_report.md"],
        "model_format": dl_persistence.MODEL_EXTENSION,
        "format_note": (
            f"The reference task spec names {dl_persistence.SPEC_EXTENSION_IN_REFERENCE}. "
            f"Keras 3 saves HDF5 but cannot reload it, so {dl_persistence.MODEL_EXTENSION} is used."
        ),
    }


def _predict(target: str, payload: dict) -> DeepPredictionResponse:
    contract = load_feature_contract()
    expected = set(contract.selected_features(target))
    given = set(payload.keys())
    if given != expected:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "feature mismatch",
                "expected": sorted(expected),
                "missing": sorted(expected - given),
                "unexpected": sorted(given - expected),
            },
        )

    model, scaler, y_scaler, spec = _load(target)
    row = np.array([[payload[f] for f in spec["features"]]], dtype="float32")
    mask = np.array(spec["numeric_mask"], dtype=bool)
    value = float(dl_training.predict(model, scaler, row, mask, y_scaler)[0])

    return DeepPredictionResponse(
        model="dnn_mlp",
        target=target,
        prediction=value,
        predicted_class=(int(value >= 0.5) if target == "target_pit_next_lap" else None),
        model_format=spec["format"],
        data_source=contract.dataset_source["source"],
    )


@router.post("/predict/laptime", response_model=DeepPredictionResponse)
def predict_laptime(request: LapTimePredictionRequest):
    return _predict("target_laptime", request.root)


@router.post("/predict/pit", response_model=DeepPredictionResponse)
def predict_pit(request: PitPredictionRequest):
    return _predict("target_pit_next_lap", request.root)
