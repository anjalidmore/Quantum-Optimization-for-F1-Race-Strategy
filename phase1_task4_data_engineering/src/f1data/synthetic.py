"""
f1data.synthetic
=================

Generates realistic, internally-consistent synthetic data matching the schemas
in :mod:`f1data.schemas`, so the entire data-engineering pipeline is runnable
without downloading the multi-gigabyte Kaggle dataset or making live FastF1
calls.

The generator deliberately injects the data-quality problems the cleaning stage
is designed to handle — missing values, duplicate rows, out-of-range outliers,
inconsistent categorical spellings, and the Ergast ``"\\N"`` null sentinel — so
the pipeline demonstrably *does something* rather than operating on already-clean
data. Everything is driven by a fixed random seed for reproducibility.

To run the pipeline on the *real* dataset instead, skip the generator and point
the loader at the genuine CSVs — the column names are identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Fixed vocabularies for realistic categorical values.
_DRIVERS = [
    ("hamilton", "HAM", "Lewis", "Hamilton", "British"),
    ("verstappen", "VER", "Max", "Verstappen", "Dutch"),
    ("leclerc", "LEC", "Charles", "Leclerc", "Monegasque"),
    ("norris", "NOR", "Lando", "Norris", "British"),
    ("alonso", "ALO", "Fernando", "Alonso", "Spanish"),
    ("russell", "RUS", "George", "Russell", "British"),
    ("sainz", "SAI", "Carlos", "Sainz", "Spanish"),
    ("perez", "PER", "Sergio", "Perez", "Mexican"),
    ("piastri", "PIA", "Oscar", "Piastri", "Australian"),
    ("gasly", "GAS", "Pierre", "Gasly", "French"),
]
_CONSTRUCTORS = [
    ("mercedes", "Mercedes", "German"),
    ("red_bull", "Red Bull", "Austrian"),
    ("ferrari", "Ferrari", "Italian"),
    ("mclaren", "McLaren", "British"),
    ("aston_martin", "Aston Martin", "British"),
    ("alpine", "Alpine", "French"),
]
_CIRCUITS = [
    ("monaco", "Circuit de Monaco", "Monte-Carlo", "Monaco", 43.7347, 7.4206, 7),
    ("silverstone", "Silverstone Circuit", "Silverstone", "UK", 52.0786, -1.0169, 153),
    ("monza", "Autodromo Nazionale Monza", "Monza", "Italy", 45.6156, 9.2811, 162),
    ("spa", "Circuit de Spa-Francorchamps", "Spa", "Belgium", 50.4372, 5.9714, 401),
    ("suzuka", "Suzuka Circuit", "Suzuka", "Japan", 34.8431, 136.5410, 45),
    ("interlagos", "Autódromo José Carlos Pace", "São Paulo", "Brazil", -23.7036, -46.6997, 785),
]
_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def _ms_to_laptime_str(ms: int) -> str:
    """Format milliseconds as Ergast 'm:ss.mmm'."""
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000.0
    return f"{minutes}:{seconds:06.3f}"


def generate_kaggle(out_dir: Path, seed: int = 42,
                    n_races: int = 24, inject_issues: bool = True) -> Dict[str, Path]:
    """
    Generate the Kaggle-style CSVs into ``out_dir``.

    Returns a mapping of table-name -> written path.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # ---- drivers ----------------------------------------------------------
    drivers = pd.DataFrame([
        {"driverId": i + 1, "driverRef": d[0], "code": d[1],
         "forename": d[2], "surname": d[3], "nationality": d[4]}
        for i, d in enumerate(_DRIVERS)
    ])
    paths["drivers"] = _save(drivers, out_dir / "drivers.csv")

    # ---- constructors -----------------------------------------------------
    constructors = pd.DataFrame([
        {"constructorId": i + 1, "constructorRef": c[0], "name": c[1],
         "nationality": c[2]}
        for i, c in enumerate(_CONSTRUCTORS)
    ])
    paths["constructors"] = _save(constructors, out_dir / "constructors.csv")

    # ---- circuits ---------------------------------------------------------
    circuits = pd.DataFrame([
        {"circuitId": i + 1, "circuitRef": c[0], "name": c[1], "location": c[2],
         "country": c[3], "lat": c[4], "lng": c[5], "alt": c[6]}
        for i, c in enumerate(_CIRCUITS)
    ])
    paths["circuits"] = _save(circuits, out_dir / "circuits.csv")

    # ---- races ------------------------------------------------------------
    race_rows = []
    for r in range(n_races):
        circuit = _CIRCUITS[r % len(_CIRCUITS)]
        race_rows.append({
            "raceId": r + 1,
            "year": 2022 + (r // 12),
            "round": (r % 12) + 1,
            "circuitId": (r % len(_CIRCUITS)) + 1,
            "name": circuit[1].split(" ")[0] + " Grand Prix",
            "date": pd.Timestamp("2022-03-20") + pd.Timedelta(days=14 * r),
        })
    races = pd.DataFrame(race_rows)
    paths["races"] = _save(races, out_dir / "races.csv")

    # ---- results, lap_times, pit_stops -----------------------------------
    results_rows: List[dict] = []
    laptime_rows: List[dict] = []
    pit_rows: List[dict] = []
    result_id = 1
    n_drivers = len(_DRIVERS)

    for _, race in races.iterrows():
        race_id = int(race["raceId"])
        race_laps = int(rng.integers(50, 72))
        # A per-race base pace and per-driver skill offsets.
        base_pace_ms = int(rng.integers(78000, 95000))
        grid = rng.permutation(n_drivers) + 1
        for di in range(n_drivers):
            driver_id = di + 1
            constructor_id = (di % len(_CONSTRUCTORS)) + 1
            skill = rng.normal(0, 400)  # ms faster/slower
            finished = rng.random() > 0.12  # ~12% DNF rate
            completed = race_laps if finished else int(rng.integers(5, race_laps))

            # lap times: base + skill + fuel-burn improvement + tyre deg noise
            fastest_ms = base_pace_ms + skill + rng.normal(0, 150)
            for lap in range(1, completed + 1):
                fuel_effect = (race_laps - lap) * 20  # heavier early = slower
                deg = (lap % 20) * 12                 # sawtooth per stint
                noise = rng.normal(0, 120)
                lt_ms = int(base_pace_ms + skill + fuel_effect + deg + noise)
                laptime_rows.append({
                    "raceId": race_id, "driverId": driver_id, "lap": lap,
                    "position": int(rng.integers(1, n_drivers + 1)),
                    "time": _ms_to_laptime_str(lt_ms), "milliseconds": lt_ms,
                })
                fastest_ms = min(fastest_ms, lt_ms)

            # pit stops: 1-3 per driver
            n_stops = int(rng.integers(1, 4))
            stop_laps = sorted(rng.choice(range(10, max(11, race_laps - 5)),
                                          size=min(n_stops, race_laps - 15),
                                          replace=False)) if race_laps > 20 else [15]
            for si, plap in enumerate(stop_laps, 1):
                dur_ms = int(rng.normal(23000, 2500))
                pit_rows.append({
                    "raceId": race_id, "driverId": driver_id, "stop": si,
                    "lap": int(plap),
                    "time": f"1{rng.integers(3,6)}:{rng.integers(10,59):02d}:{rng.integers(10,59):02d}",
                    "duration": f"{dur_ms/1000:.3f}", "milliseconds": dur_ms,
                })

            position = int(np.where(grid == (di + 1))[0][0]) + 1 if finished else None
            points_table = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6,
                            8: 4, 9: 2, 10: 1}
            results_rows.append({
                "resultId": result_id, "raceId": race_id, "driverId": driver_id,
                "constructorId": constructor_id, "grid": int(grid[di]),
                "position": position,
                "points": points_table.get(position, 0) if position else 0,
                "laps": completed,
                "statusId": 1 if finished else int(rng.integers(2, 20)),
                "fastestLapTime": _ms_to_laptime_str(int(fastest_ms)),
                "rank": int(rng.integers(1, n_drivers + 1)),
            })
            result_id += 1

    results = pd.DataFrame(results_rows)
    lap_times = pd.DataFrame(laptime_rows)
    pit_stops = pd.DataFrame(pit_rows)

    if inject_issues:
        results = _inject_issues(results, rng, numeric_cols=["points", "grid"],
                                 cat_cols=[])
        lap_times = _inject_issues(lap_times, rng,
                                   numeric_cols=["milliseconds"], cat_cols=[])
        pit_stops = _inject_issues(pit_stops, rng,
                                   numeric_cols=["milliseconds"], cat_cols=[])

    paths["results"] = _save(results, out_dir / "results.csv")
    paths["lap_times"] = _save(lap_times, out_dir / "lap_times.csv")
    paths["pit_stops"] = _save(pit_stops, out_dir / "pit_stops.csv")
    return paths


def generate_fastf1_laps(out_dir: Path, seed: int = 7,
                         n_drivers: int = 10, n_laps: int = 55,
                         inject_issues: bool = True) -> Path:
    """Generate a FastF1-style laps CSV into ``out_dir``."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    base_air, base_track = 24.0, 38.0
    for di in range(n_drivers):
        code = _DRIVERS[di][1]
        team = _CONSTRUCTORS[di % len(_CONSTRUCTORS)][1]
        skill = rng.normal(0, 0.4)
        # 2-3 stints with different compounds
        n_stints = int(rng.integers(2, 4))
        stint_bounds = np.linspace(0, n_laps, n_stints + 1).astype(int)
        for stint in range(n_stints):
            compound = _COMPOUNDS[stint % len(_COMPOUNDS)]
            start, end = stint_bounds[stint], stint_bounds[stint + 1]
            for i, lap in enumerate(range(start + 1, end + 1), 1):
                tyre_life = i
                base = 90.5 + skill + {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.4}[compound]
                deg = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}[compound] * tyre_life
                lap_time = base + deg + rng.normal(0, 0.15)
                s1 = lap_time * 0.30 + rng.normal(0, 0.03)
                s2 = lap_time * 0.42 + rng.normal(0, 0.03)
                s3 = lap_time - s1 - s2
                rows.append({
                    "Driver": code, "Team": team, "LapNumber": lap,
                    "LapTime": round(lap_time, 3), "Stint": stint + 1,
                    "Compound": compound, "TyreLife": tyre_life,
                    "FreshTyre": "True" if i == 1 else "False",
                    "Sector1Time": round(s1, 3), "Sector2Time": round(s2, 3),
                    "Sector3Time": round(s3, 3),
                    "SpeedFL": round(rng.normal(290, 8), 1),
                    "SpeedST": round(rng.normal(320, 10), 1),
                    "TrackStatus": "1",
                    "AirTemp": round(base_air + rng.normal(0, 1.5), 1),
                    "TrackTemp": round(base_track + rng.normal(0, 2.5), 1),
                    "Humidity": round(rng.uniform(35, 70), 1),
                    "Rainfall": "False",
                    "WindSpeed": round(rng.uniform(0, 6), 1),
                    "IsPersonalBest": "True" if rng.random() < 0.05 else "False",
                })
    laps = pd.DataFrame(rows)
    if inject_issues:
        laps = _inject_issues(laps, rng, numeric_cols=["LapTime", "TrackTemp"],
                              cat_cols=["Compound"])
    return _save(laps, out_dir / "fastf1_laps.csv")


# --------------------------------------------------------------------------- #
# Issue injection & IO helpers
# --------------------------------------------------------------------------- #

def _inject_issues(df: pd.DataFrame, rng: np.random.Generator,
                   numeric_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Inject missing values, duplicates, outliers, and inconsistent categories."""
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    # 1. Missing values (Ergast uses the literal "\N"; also blank NaNs).
    for col in numeric_cols:
        if col in df.columns:
            idx = rng.choice(n, size=max(1, n // 50), replace=False)
            df.loc[df.index[idx], col] = np.nan

    # 2. Duplicate rows (~1%).
    dup_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # 3. Outliers in a numeric column.
    if numeric_cols and numeric_cols[0] in df.columns:
        col = numeric_cols[0]
        out_idx = rng.choice(len(df), size=max(1, len(df) // 200), replace=False)
        df.loc[df.index[out_idx], col] = df[col].astype(float).max() * 6

    # 4. Inconsistent categorical spellings (case / whitespace).
    for col in cat_cols:
        if col in df.columns:
            m_idx = rng.choice(len(df), size=max(1, len(df) // 50), replace=False)
            df.loc[df.index[m_idx], col] = df.loc[df.index[m_idx], col].astype(str).str.lower()

    return df


def _save(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def generate_all(raw_dir: Path, seed: int = 42) -> Dict[str, Path]:
    """Generate the full synthetic dataset (Kaggle CSVs + FastF1 laps)."""
    paths = generate_kaggle(raw_dir, seed=seed)
    paths["fastf1_laps"] = generate_fastf1_laps(raw_dir, seed=seed + 1)
    return paths
