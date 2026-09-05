from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.routers.data import _options
from app.api.schemas import RaceStateRequest
from app.services.model_cache import get_model_cache
from app.services.strategy_service import run_strategy_analysis

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


def _validate_against_known_options(race_state: RaceStateRequest) -> None:
    options = _options()
    if race_state.driver.strip().upper() not in {d.upper() for d in options.drivers}:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown driver {race_state.driver!r}. Valid options: {options.drivers}",
        )
    if race_state.team.strip().upper() not in {t.upper() for t in options.teams}:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown team {race_state.team!r}. Valid options: {options.teams}",
        )
    if race_state.tyre_compound.strip().upper() not in {c.upper() for c in options.compounds}:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown tyre compound {race_state.tyre_compound!r}. Valid options: {options.compounds}",
        )


def _validate_model_choice(target: str, model_name: str | None) -> None:
    if model_name is None:
        return
    cache = get_model_cache()
    registry = cache.registry()
    # Deep models are excluded: this endpoint serves Task 6 sklearn pipelines
    # through ModelCache, which cannot load a .keras archive. They are reachable
    # via /api/dl/predict/* instead.
    valid = {
        m["model_name"]
        for m in (registry or {}).get("models", [])
        if m["target"] == target and m["artifact"] and m.get("family") != "deep"
    }
    if model_name not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Model {model_name!r} is not an available trained model for {target!r}. Available: {sorted(valid)}",
        )


@router.post("/predict")
def predict_strategy(race_state: RaceStateRequest):
    _validate_against_known_options(race_state)
    _validate_model_choice("target_laptime", race_state.laptime_model)
    _validate_model_choice("target_pit_next_lap", race_state.pit_model)
    return run_strategy_analysis(
        race_state,
        laptime_model=race_state.laptime_model,
        pit_model=race_state.pit_model,
        explain=race_state.explain,
    )
