from __future__ import annotations

from fastapi import APIRouter

from app.intelligence.ml.regression import xgboost_available, xgboost_unavailable_reason
from app.services.model_cache import get_model_cache

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    cache = get_model_cache()
    registry = cache.registry()
    return {
        "status": "ok",
        "models_trained": registry is not None,
        "model_count": len(registry["models"]) if registry else 0,
        "xgboost_available": xgboost_available(),
        "xgboost_status": None if xgboost_available() else xgboost_unavailable_reason(),
    }
