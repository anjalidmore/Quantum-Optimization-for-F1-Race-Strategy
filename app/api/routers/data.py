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

from fastapi import APIRouter, HTTPException

from app.api.schemas import DataOptionsResponse
from app.intelligence.features.contract import load_feature_contract, load_feature_matrix

router = APIRouter(prefix="/api/data", tags=["data"])


@lru_cache(maxsize=1)
def _options() -> DataOptionsResponse:
    df = load_feature_matrix()
    contract = load_feature_contract()
    return DataOptionsResponse(
        drivers=sorted(df["Driver"].dropna().unique().tolist()),
        teams=sorted(df["Team"].dropna().unique().tolist()),
        compounds=sorted(df["Compound"].dropna().unique().tolist()),
        total_laps_hint=int(df["LapNumber"].max()),
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
