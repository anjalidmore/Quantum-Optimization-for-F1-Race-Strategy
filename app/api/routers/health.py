from __future__ import annotations

from fastapi import APIRouter

from app.core.paths import DL_MODELS_DIR, XAI_RESULTS_JSON
from app.intelligence.dl import persistence as dl_persistence
from app.intelligence.ml.regression import xgboost_available, xgboost_unavailable_reason
from app.services.model_cache import get_model_cache

router = APIRouter(prefix="/api", tags=["health"])

_TARGETS = ("target_laptime", "target_pit_next_lap")


def _ml_status() -> dict:
    registry = get_model_cache().registry()
    models = (registry or {}).get("models", [])
    out = {}
    for target in _TARGETS:
        best = next((m["model_name"] for m in models if m["task"] == target and m["is_selected_best"]), None)
        out[target] = {"trained": best is not None, "selected_model": best}
    return out, registry


def _dl_status() -> dict:
    out = {}
    for target in _TARGETS:
        d = dl_persistence.target_dir(DL_MODELS_DIR, target)
        model_path = d / dl_persistence.MODEL_FILENAME
        spec_path = d / "model_spec.json"
        out[target] = {"trained": model_path.exists() and spec_path.exists()}
    return out


@router.get("/health")
def health():
    ml_status, registry = _ml_status()
    dl_status = _dl_status()
    return {
        "status": "ok",
        "models_trained": registry is not None,
        "model_count": len(registry["models"]) if registry else 0,
        "xgboost_available": xgboost_available(),
        "xgboost_status": None if xgboost_available() else xgboost_unavailable_reason(),
        "models": {
            "ml": ml_status,
            "dl": dl_status,
            "xai_available": XAI_RESULTS_JSON.exists(),
        },
    }
