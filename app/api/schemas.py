"""Pydantic request/response models for the Task 6 ML API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class LapTimePredictionRequest(RootModel[dict[str, float]]):
    """A flat dict of Task 5 regression feature name -> value, e.g.
    ``{"tyre_life": 12, "gap_roll3_mean": 0.15, ...}``. Must contain exactly
    the features listed by ``GET /api/ml/models`` for the laptime task."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "tyre_life": 12,
            "gap_roll3_mean": 0.15,
            "gap_expanding": 0.10,
            "field_median_lag1": 89.4,
            "driver_sai": 0,
            "race_progress": 0.55,
        }
    })


class PitPredictionRequest(RootModel[dict[str, float]]):
    """A flat dict of Task 5 classification feature name -> value."""


class LapTimePredictionResponse(BaseModel):
    model: str
    prediction: float
    target: str = "target_laptime"
    unit: str = "seconds"
    model_version: str | None = None
    # No default: must be set explicitly from the live feature contract
    # (contract.dataset_source["source"]) — never assumed "synthetic".
    data_source: str


class PitPredictionResponse(BaseModel):
    model: str
    probability_pit: float
    predicted_class: int
    target: str = "target_pit_next_lap"
    model_version: str | None = None
    data_source: str


class RaceStateRequest(BaseModel):
    driver: str = Field(..., description="Driver code, e.g. 'HAM'. Must be one of GET /api/data/options.drivers.")
    team: str = Field(..., description="Team name, e.g. 'Aston Martin'. Must be one of GET /api/data/options.teams.")
    current_lap: int = Field(..., ge=1)
    total_laps: int = Field(..., ge=1)
    tyre_compound: str = Field(..., description="SOFT | MEDIUM | HARD | INTERMEDIATE | WET")
    tyre_age: int = Field(..., ge=0)
    track_temperature: float
    weather: str = Field("dry", description="dry | damp | wet | extreme")
    fuel_kg: float = Field(100.0, ge=0)
    track_status: str = Field("GREEN", description="GREEN | YELLOW | SC | VSC | RED")
    current_position: int = Field(5, ge=1, le=24)
    # Optional explicit model choice per task; omit (or null) to use the
    # registry's selected-best model for that task.
    laptime_model: str | None = None
    pit_model: str | None = None
    # Opt-in Task 8 explanation. Defaults to false so the response shape and
    # latency are unchanged for every existing caller; SHAP sampling costs
    # roughly a second per target, which a pit-wall caller may not want to pay.
    explain: bool = Field(
        False,
        description="Attach a Task 8 explanation (SHAP factors, trust score, "
                    "plain-English narrative) for this prediction's feature rows.",
    )

    @model_validator(mode="after")
    def _check_race_consistency(self) -> "RaceStateRequest":
        if self.current_lap > self.total_laps:
            raise ValueError(f"current_lap ({self.current_lap}) cannot exceed total_laps ({self.total_laps}).")
        if self.tyre_age > self.current_lap:
            raise ValueError(f"tyre_age ({self.tyre_age}) cannot exceed current_lap ({self.current_lap}).")
        return self


class DataOptionsResponse(BaseModel):
    """Real, dataset-derived choices for the race-strategy simulator's
    dropdowns — never a hard-coded list that could drift from what the
    trained models actually saw."""

    drivers: list[str]
    teams: list[str]
    compounds: list[str]
    total_laps_hint: int
    track_temperature_range: dict  # {min, mean, max} — the real range this session's model was trained on
    dataset_source: dict


class DeepPredictionResponse(BaseModel):
    """Task 7 prediction from a Keras network.

    ``prediction`` is lap time in seconds for the regression target and a pit
    probability in [0, 1] for the classification target; ``predicted_class`` is
    populated only for the latter. ``model_format`` is surfaced because the
    reference task spec names ``.h5`` while Keras 3 requires ``.keras``.
    """

    model: str
    target: str
    prediction: float
    predicted_class: int | None = None
    model_format: str
    data_source: str

    model_config = {"protected_namespaces": ()}
