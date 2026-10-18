"""
app.api.routers.rai
===================

Responsible-AI endpoints for the dashboard's RAI tab: group-wise performance
(fairness), prediction confidence and uncertainty for the Simple, ML and DNN
models on the shared holdout. Read-only inference over persisted models; see
``app.services.rai_service`` for what is and is not computed, and why.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import rai_service

router = APIRouter(prefix="/api/rai", tags=["responsible-ai"])

TARGETS = ("target_laptime", "target_pit_next_lap")


@router.get("/summary")
def get_summary(target: str = Query("target_laptime")):
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"Unknown target {target!r}; expected one of {list(TARGETS)}")
    try:
        return rai_service.summary(target)
    except rai_service.RaiUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
