"""
f1data.schemas
==============

Canonical schemas for the two data sources this pipeline consumes.

1. **Historical Kaggle dataset** — "Formula 1 World Championship (1950–2020)"
   (rohanrao/formula-1-world-championship-1950-2020), an Ergast-derived set of
   CSVs. The column names below match the real files exactly, so the loader
   works unchanged on a genuine download.

2. **FastF1 laps** — the tabular lap/stint/telemetry-summary frame produced by
   ``fastf1``'s ``session.laps``. Again the column names match the library.

These schemas serve three purposes: they document the expected inputs, they
drive the synthetic-data generator (so the whole pipeline is runnable without
the multi-gigabyte download), and they let the loader validate that a real file
has the columns the pipeline relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class ColKind(str, Enum):
    """Coarse column kind, used by cleaning/encoding/scaling stages."""

    ID = "id"                 # identifier / key (not a model feature)
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    DATE = "date"
    TIME = "time"             # duration-like (e.g. lap time)
    TEXT = "text"


@dataclass(frozen=True)
class Column:
    name: str
    kind: ColKind
    description: str
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    """Schema for one CSV / dataframe."""

    name: str
    source: str  # "kaggle" or "fastf1"
    columns: Tuple[Column, ...]
    primary_key: Tuple[str, ...] = ()

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def by_kind(self, kind: ColKind) -> List[str]:
        return [c.name for c in self.columns if c.kind is kind]


_C = Column
_K = ColKind

# --------------------------------------------------------------------------- #
# Kaggle CSV schemas (subset of the tables most relevant to strategy analysis)
# --------------------------------------------------------------------------- #

RACES = TableSchema(
    "races", "kaggle",
    (
        _C("raceId", _K.ID, "Race identifier.", nullable=False),
        _C("year", _K.NUMERIC, "Season year.", nullable=False),
        _C("round", _K.NUMERIC, "Round number in the season."),
        _C("circuitId", _K.ID, "Circuit identifier."),
        _C("name", _K.CATEGORICAL, "Race / Grand Prix name."),
        _C("date", _K.DATE, "Race date."),
    ),
    primary_key=("raceId",),
)

DRIVERS = TableSchema(
    "drivers", "kaggle",
    (
        _C("driverId", _K.ID, "Driver identifier.", nullable=False),
        _C("driverRef", _K.CATEGORICAL, "Stable driver reference."),
        _C("code", _K.CATEGORICAL, "Three-letter code."),
        _C("forename", _K.TEXT, "First name."),
        _C("surname", _K.TEXT, "Last name."),
        _C("nationality", _K.CATEGORICAL, "Nationality."),
    ),
    primary_key=("driverId",),
)

CONSTRUCTORS = TableSchema(
    "constructors", "kaggle",
    (
        _C("constructorId", _K.ID, "Constructor identifier.", nullable=False),
        _C("constructorRef", _K.CATEGORICAL, "Stable reference."),
        _C("name", _K.CATEGORICAL, "Constructor name."),
        _C("nationality", _K.CATEGORICAL, "Nationality."),
    ),
    primary_key=("constructorId",),
)

CIRCUITS = TableSchema(
    "circuits", "kaggle",
    (
        _C("circuitId", _K.ID, "Circuit identifier.", nullable=False),
        _C("circuitRef", _K.CATEGORICAL, "Stable reference."),
        _C("name", _K.CATEGORICAL, "Circuit name."),
        _C("location", _K.CATEGORICAL, "City / locality."),
        _C("country", _K.CATEGORICAL, "Country."),
        _C("lat", _K.NUMERIC, "Latitude."),
        _C("lng", _K.NUMERIC, "Longitude."),
        _C("alt", _K.NUMERIC, "Altitude (m)."),
    ),
    primary_key=("circuitId",),
)

RESULTS = TableSchema(
    "results", "kaggle",
    (
        _C("resultId", _K.ID, "Result identifier.", nullable=False),
        _C("raceId", _K.ID, "Race identifier."),
        _C("driverId", _K.ID, "Driver identifier."),
        _C("constructorId", _K.ID, "Constructor identifier."),
        _C("grid", _K.NUMERIC, "Starting grid position."),
        _C("position", _K.NUMERIC, "Finishing position (nullable if DNF)."),
        _C("points", _K.NUMERIC, "Points scored."),
        _C("laps", _K.NUMERIC, "Laps completed."),
        _C("statusId", _K.ID, "Finish-status identifier."),
        _C("fastestLapTime", _K.TIME, "Fastest lap time (m:ss.mmm)."),
        _C("rank", _K.NUMERIC, "Fastest-lap rank."),
    ),
    primary_key=("resultId",),
)

PIT_STOPS = TableSchema(
    "pit_stops", "kaggle",
    (
        _C("raceId", _K.ID, "Race identifier.", nullable=False),
        _C("driverId", _K.ID, "Driver identifier.", nullable=False),
        _C("stop", _K.NUMERIC, "Ordinal stop number.", nullable=False),
        _C("lap", _K.NUMERIC, "Lap of the stop."),
        _C("time", _K.TIME, "Time of day of the stop."),
        _C("duration", _K.TIME, "Stationary + pit-lane duration (s)."),
        _C("milliseconds", _K.NUMERIC, "Duration in milliseconds."),
    ),
    primary_key=("raceId", "driverId", "stop"),
)

LAP_TIMES = TableSchema(
    "lap_times", "kaggle",
    (
        _C("raceId", _K.ID, "Race identifier.", nullable=False),
        _C("driverId", _K.ID, "Driver identifier.", nullable=False),
        _C("lap", _K.NUMERIC, "Lap number.", nullable=False),
        _C("position", _K.NUMERIC, "Position on that lap."),
        _C("time", _K.TIME, "Lap time (m:ss.mmm)."),
        _C("milliseconds", _K.NUMERIC, "Lap time in milliseconds."),
    ),
    primary_key=("raceId", "driverId", "lap"),
)

# --------------------------------------------------------------------------- #
# FastF1 laps schema (subset of session.laps columns most relevant here)
# --------------------------------------------------------------------------- #

FASTF1_LAPS = TableSchema(
    "fastf1_laps", "fastf1",
    (
        _C("Driver", _K.CATEGORICAL, "Three-letter driver code.", nullable=False),
        _C("Team", _K.CATEGORICAL, "Team name."),
        _C("LapNumber", _K.NUMERIC, "Lap index within the session."),
        _C("LapTime", _K.TIME, "Lap time (seconds)."),
        _C("Stint", _K.NUMERIC, "Stint number."),
        _C("Compound", _K.CATEGORICAL, "Tyre compound."),
        _C("TyreLife", _K.NUMERIC, "Laps on the current set."),
        _C("FreshTyre", _K.CATEGORICAL, "Whether the set was fresh."),
        _C("Sector1Time", _K.TIME, "Sector 1 time (s)."),
        _C("Sector2Time", _K.TIME, "Sector 2 time (s)."),
        _C("Sector3Time", _K.TIME, "Sector 3 time (s)."),
        _C("SpeedFL", _K.NUMERIC, "Speed at the finish line (km/h)."),
        _C("SpeedST", _K.NUMERIC, "Speed-trap speed (km/h)."),
        _C("TrackStatus", _K.CATEGORICAL, "Track-status flag code."),
        _C("AirTemp", _K.NUMERIC, "Air temperature (°C)."),
        _C("TrackTemp", _K.NUMERIC, "Track temperature (°C)."),
        _C("Humidity", _K.NUMERIC, "Relative humidity (%)."),
        _C("Rainfall", _K.CATEGORICAL, "Whether rain was recorded."),
        _C("WindSpeed", _K.NUMERIC, "Wind speed (m/s)."),
        _C("IsPersonalBest", _K.CATEGORICAL, "Whether it was a personal best."),
    ),
    primary_key=("Driver", "LapNumber"),
)


ALL_SCHEMAS: Dict[str, TableSchema] = {
    s.name: s
    for s in (RACES, DRIVERS, CONSTRUCTORS, CIRCUITS, RESULTS, PIT_STOPS,
              LAP_TIMES, FASTF1_LAPS)
}

KAGGLE_SCHEMAS: Dict[str, TableSchema] = {
    n: s for n, s in ALL_SCHEMAS.items() if s.source == "kaggle"
}
