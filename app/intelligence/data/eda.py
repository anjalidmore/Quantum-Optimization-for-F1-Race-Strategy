"""
f1data.eda
==========

Exploratory data analysis: statistical summaries, correlation analysis, a
data-quality scorecard, and the domain-specific analyses requested for Task 4
(driver, circuit, constructor, tyre, pit-stop, weather, lap-time, safety-car and
season analysis).

Each function returns plain pandas objects / dataclasses so the results can be
rendered into Markdown reports and figures without coupling to the presentation
layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Statistical summary
# --------------------------------------------------------------------------- #

def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return an extended per-numeric-column summary: count, mean, std, min, 25%,
    50%, 75%, max, skew and kurtosis.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    desc["missing"] = df[numeric.columns].isna().sum()
    return desc.round(4)


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #

@dataclass
class CorrelationResult:
    matrix: pd.DataFrame
    top_pairs: List[tuple]  # (col_a, col_b, correlation), strongest first


def correlation_analysis(df: pd.DataFrame, method: str = "pearson",
                         top_n: int = 15) -> CorrelationResult:
    """Compute the correlation matrix and the strongest off-diagonal pairs."""
    numeric = df.select_dtypes(include=[np.number])
    # Drop constant columns (undefined correlation).
    numeric = numeric.loc[:, numeric.std(ddof=0) > 0]
    if numeric.shape[1] < 2:
        return CorrelationResult(pd.DataFrame(), [])
    matrix = numeric.corr(method=method)
    pairs: List[tuple] = []
    cols = matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = matrix.iloc[i, j]
            if pd.notna(corr):
                pairs.append((cols[i], cols[j], float(corr)))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return CorrelationResult(matrix.round(4), pairs[:top_n])


# --------------------------------------------------------------------------- #
# Data-quality scorecard
# --------------------------------------------------------------------------- #

@dataclass
class QualityReport:
    table: str
    n_rows: int
    n_cols: int
    completeness: float          # fraction of non-null cells
    duplicate_rows: int
    per_column_missing: Dict[str, float]
    constant_columns: List[str]
    quality_score: float         # 0-100 composite


def data_quality(df: pd.DataFrame, table: str) -> QualityReport:
    """Compute a data-quality scorecard for ``df``."""
    n_rows, n_cols = df.shape
    total_cells = max(1, n_rows * n_cols)
    n_missing = int(df.isna().sum().sum())
    completeness = 1.0 - n_missing / total_cells
    duplicate_rows = int(df.duplicated().sum())
    per_col_missing = (df.isna().mean()).round(4).to_dict()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    # Composite score: weight completeness, uniqueness, and non-constancy.
    uniqueness = 1.0 - (duplicate_rows / max(1, n_rows))
    non_constant = 1.0 - (len(constant_cols) / max(1, n_cols))
    score = 100.0 * (0.6 * completeness + 0.25 * uniqueness + 0.15 * non_constant)

    return QualityReport(
        table=table, n_rows=n_rows, n_cols=n_cols,
        completeness=round(completeness, 4), duplicate_rows=duplicate_rows,
        per_column_missing=per_col_missing, constant_columns=constant_cols,
        quality_score=round(score, 2),
    )


# --------------------------------------------------------------------------- #
# Domain-specific analyses
# --------------------------------------------------------------------------- #

def driver_analysis(results: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Points, wins, podiums and average finishing position per driver."""
    df = results.merge(drivers[["driverId", "surname", "code"]], on="driverId",
                       how="left")
    agg = df.groupby(["driverId", "surname", "code"], dropna=False).agg(
        races=("resultId", "count"),
        total_points=("points", "sum"),
        avg_points=("points", "mean"),
        wins=("position", lambda s: int((s == 1).sum())),
        podiums=("position", lambda s: int((s <= 3).sum())),
        avg_finish=("position", "mean"),
    ).reset_index()
    return agg.sort_values("total_points", ascending=False).round(3)


def constructor_analysis(results: pd.DataFrame,
                         constructors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate constructor performance."""
    df = results.merge(constructors[["constructorId", "name"]],
                       on="constructorId", how="left")
    agg = df.groupby(["constructorId", "name"], dropna=False).agg(
        entries=("resultId", "count"),
        total_points=("points", "sum"),
        avg_points=("points", "mean"),
        wins=("position", lambda s: int((s == 1).sum())),
    ).reset_index()
    return agg.sort_values("total_points", ascending=False).round(3)


def circuit_analysis(results: pd.DataFrame, races: pd.DataFrame,
                     circuits: pd.DataFrame) -> pd.DataFrame:
    """Average points and field attrition per circuit."""
    df = results.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
    df = df.merge(circuits[["circuitId", "name", "country"]], on="circuitId",
                  how="left")
    agg = df.groupby(["circuitId", "name", "country"], dropna=False).agg(
        entries=("resultId", "count"),
        finishers=("position", lambda s: int(s.notna().sum())),
        avg_points=("points", "mean"),
    ).reset_index()
    agg["dnf_rate"] = (1 - agg["finishers"] / agg["entries"]).round(3)
    return agg.round(3)


def pit_stop_analysis(pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Pit-stop duration statistics and stops-per-race distribution."""
    ps = pit_stops.copy()
    ps["duration_s"] = pd.to_numeric(ps.get("duration"), errors="coerce")
    ps.loc[ps["duration_s"].isna(), "duration_s"] = ps["milliseconds"] / 1000.0
    summary = pd.DataFrame({
        "metric": ["count", "mean_s", "median_s", "std_s", "min_s", "max_s",
                   "avg_stops_per_driver_race"],
        "value": [
            int(ps["duration_s"].notna().sum()),
            round(ps["duration_s"].mean(), 3),
            round(ps["duration_s"].median(), 3),
            round(ps["duration_s"].std(), 3),
            round(ps["duration_s"].min(), 3),
            round(ps["duration_s"].max(), 3),
            round(ps.groupby(["raceId", "driverId"])["stop"].max().mean(), 3),
        ],
    })
    return summary


def tyre_analysis(fastf1_laps: pd.DataFrame) -> pd.DataFrame:
    """Per-compound pace and degradation from FastF1 laps."""
    df = fastf1_laps.copy()
    df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df = df.dropna(subset=["LapTime", "Compound"])
    rows = []
    for compound, grp in df.groupby("Compound"):
        # Degradation slope: seconds per lap of tyre life (simple linear fit).
        slope = np.nan
        g = grp.dropna(subset=["TyreLife"])
        if len(g) >= 2 and g["TyreLife"].std(ddof=0) > 0:
            slope = float(np.polyfit(g["TyreLife"], g["LapTime"], 1)[0])
        rows.append({
            "compound": compound,
            "n_laps": len(grp),
            "mean_laptime_s": round(grp["LapTime"].mean(), 3),
            "best_laptime_s": round(grp["LapTime"].min(), 3),
            "deg_s_per_lap": round(slope, 4) if slope == slope else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_laptime_s")


def lap_time_analysis(lap_times: pd.DataFrame) -> pd.DataFrame:
    """Lap-time distribution statistics (milliseconds -> seconds)."""
    lt = pd.to_numeric(lap_times["milliseconds"], errors="coerce") / 1000.0
    lt = lt.dropna()
    return pd.DataFrame({
        "metric": ["count", "mean_s", "median_s", "std_s", "p05_s", "p95_s"],
        "value": [
            int(lt.count()), round(lt.mean(), 3), round(lt.median(), 3),
            round(lt.std(), 3), round(lt.quantile(0.05), 3),
            round(lt.quantile(0.95), 3),
        ],
    })


def weather_analysis(fastf1_laps: pd.DataFrame) -> pd.DataFrame:
    """Summary of the weather channels present in FastF1 laps."""
    cols = ["AirTemp", "TrackTemp", "Humidity", "WindSpeed"]
    present = [c for c in cols if c in fastf1_laps.columns]
    rows = []
    for c in present:
        s = pd.to_numeric(fastf1_laps[c], errors="coerce").dropna()
        rows.append({"channel": c, "mean": round(s.mean(), 2),
                     "min": round(s.min(), 2), "max": round(s.max(), 2),
                     "std": round(s.std(), 2)})
    return pd.DataFrame(rows)


def season_analysis(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Points scored and races held per season."""
    df = results.merge(races[["raceId", "year"]], on="raceId", how="left")
    agg = df.groupby("year", dropna=False).agg(
        races=("raceId", "nunique"),
        total_points=("points", "sum"),
        avg_points_per_entry=("points", "mean"),
    ).reset_index()
    return agg.round(3)


def safety_car_analysis(fastf1_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Track-status distribution as a proxy for safety-car / caution periods.

    FastF1 encodes track status numerically; here we report the frequency of
    each status code across laps (status '4' = SC, '6'/'7' = VSC in the real
    feed; the synthetic data is predominantly green '1').
    """
    if "TrackStatus" not in fastf1_laps.columns:
        return pd.DataFrame({"status": [], "laps": []})
    counts = fastf1_laps["TrackStatus"].astype(str).value_counts()
    return pd.DataFrame({"status": counts.index, "laps": counts.values})
