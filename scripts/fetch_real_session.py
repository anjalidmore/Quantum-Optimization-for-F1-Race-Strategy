#!/usr/bin/env python3
"""
fetch_real_session.py
======================

Replaces the synthetic `fastf1_laps.csv` with a real session's lap data,
fetched live via the `fastf1` package, formatted to match
`app.intelligence.data.schemas.FASTF1_LAPS` exactly so the existing Task 4
cleaning pipeline (`scripts/run_eda.py`) runs on it unchanged.

Only `fastf1_laps.csv` is replaced. The other Kaggle-style tables
(races/drivers/constructors/circuits/results/pit_stops/lap_times.csv) are
used by Task 4's *other* EDA analyses (driver/constructor/season stats), not
by Task 5/6 — the feature-engineering contract is built exclusively from
`fastf1_laps_clean.csv` (see `feature_metadata.json: source_dataset`) — so
replacing just this one file is sufficient to make Task 5/6 train on a real
Grand Prix.

Usage
-----
    python scripts/fetch_real_session.py --year 2023 --event Bahrain --session R
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from app.core.paths import DATA_RAW_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("fetch_real_session")

# Columns app.intelligence.data.schemas.FASTF1_LAPS expects, in order.
_SCHEMA_COLUMNS = [
    "Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
    "FreshTyre", "Sector1Time", "Sector2Time", "Sector3Time", "SpeedFL", "SpeedST",
    "TrackStatus", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed",
    "IsPersonalBest",
]

_TIME_COLS = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds()


def fetch(year: int, event: str, session_type: str, cache_dir: Path) -> pd.DataFrame:
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    log.info("Loading %s %s (%s) via FastF1 ...", year, event, session_type)
    session = fastf1.get_session(year, event, session_type)
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps.copy()
    log.info("Fetched %d raw laps for %d drivers.", len(laps), laps["Driver"].nunique())

    # Attach the nearest weather sample to every lap (fastf1's own helper).
    weather = laps.get_weather_data()
    for col in ("AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"):
        laps[col] = weather[col].to_numpy()

    for col in _TIME_COLS:
        laps[col] = _timedelta_to_seconds(laps[col])

    out = laps[_SCHEMA_COLUMNS].copy()

    # Drop laps with no recorded lap time (formation lap, red-flag laps, etc.)
    # rather than letting Task 4 median-impute a fabricated race-defining value.
    before = len(out)
    out = out.dropna(subset=["LapTime"]).reset_index(drop=True)
    log.info("Dropped %d lap(s) with no recorded LapTime (formation/red-flag laps).", before - len(out))

    # Match the synthetic generator's string formatting for boolean-like
    # categorical columns so downstream normalisation behaves identically.
    for col in ("FreshTyre", "Rainfall", "IsPersonalBest"):
        out[col] = out[col].map({True: "True", False: "False"}).fillna(out[col])
    out["TrackStatus"] = out["TrackStatus"].astype(str)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a real FastF1 session into data/raw/fastf1_laps.csv")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", type=str, required=True, help="Event name or round number, e.g. 'Bahrain' or 1")
    parser.add_argument("--session", type=str, default="R", help="Session identifier: R, Q, FP1, FP2, FP3, S")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/fastf1_cache"))
    parser.add_argument("--output", type=Path, default=DATA_RAW_DIR / "fastf1_laps.csv")
    args = parser.parse_args()

    try:
        df = fetch(args.year, args.event, args.session, args.cache_dir)
    except Exception as exc:
        log.error("Failed to fetch real session data: %s: %s", type(exc).__name__, exc)
        log.error("This requires network access to F1's live-timing archive. "
                   "The pipeline continues to work on the existing synthetic data if this fails.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Wrote %d real laps -> %s", len(df), args.output)

    # Record real provenance so downstream stages (Task 4 cleaning, the Task 5
    # notebook, reports, the API, the frontend) can say "real" vs "synthetic"
    # truthfully instead of assuming synthetic by default.
    marker = args.output.parent / ".data_source.json"
    marker.write_text(json.dumps({
        "source": "real_fastf1",
        "year": args.year,
        "event": args.event,
        "session": args.session,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_laps": len(df),
        "n_drivers": int(df["Driver"].nunique()),
    }, indent=2))
    log.info("Wrote data-source marker -> %s", marker)
    log.info("Next: python scripts/build_all.py --force   (re-cleans data/raw/, NEVER pass --regenerate-synthetic here)")
    log.info("      then re-run docs/notebooks/task5_feature_engineering.ipynb against the new clean CSV")
    log.info("      then python scripts/build_all.py --force   (retrains Task 6 on the new feature matrix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
