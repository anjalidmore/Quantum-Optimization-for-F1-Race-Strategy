"""
app.api.routers.data
=======================

Real, dataset-derived choices for the frontend's dropdowns (driver, team,
tyre compound) — the whole reason the earlier version of the race-strategy
simulator's driver/team fields had no effect on the prediction was that they
accepted arbitrary free text that usually didn't match anything in the
training data. Serving the real values here, and having the frontend use
them as an actual dropdown, fixes that at the source.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.api.schemas import DataOptionsResponse
from app.core.paths import FASTF1_LAPS_CLEAN_CSV
from app.intelligence.features.contract import load_feature_contract, load_feature_matrix

router = APIRouter(prefix="/api/data", tags=["data"])


def _track_temperature_range() -> dict:
    """The real TrackTemp range the model was trained on — used by the
    frontend to build demo scenarios that don't silently ask the model to
    extrapolate wildly beyond conditions it ever saw (e.g. a "hot track"
    preset hard-coded to 48°C when the actual session never exceeded 31°C)."""
    if FASTF1_LAPS_CLEAN_CSV.exists():
        df = pd.read_csv(FASTF1_LAPS_CLEAN_CSV)
        if "TrackTemp" in df.columns:
            return {"min": float(df["TrackTemp"].min()), "mean": float(df["TrackTemp"].mean()), "max": float(df["TrackTemp"].max())}
    return {"min": 25.0, "mean": 35.0, "max": 45.0}


@lru_cache(maxsize=1)
def _options() -> DataOptionsResponse:
    df = load_feature_matrix()
    contract = load_feature_contract()
    return DataOptionsResponse(
        drivers=sorted(df["Driver"].dropna().unique().tolist()),
        teams=sorted(df["Team"].dropna().unique().tolist()),
        compounds=sorted(df["Compound"].dropna().unique().tolist()),
        total_laps_hint=int(df["LapNumber"].max()),
        track_temperature_range=_track_temperature_range(),
        dataset_source=contract.dataset_source,
    )


@router.get("/options", response_model=DataOptionsResponse)
def get_options():
    try:
        return _options()
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Task 5 feature matrix not available: {exc}. Run the training pipeline first.",
        ) from exc
