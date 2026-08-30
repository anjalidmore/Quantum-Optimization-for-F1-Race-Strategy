"""
Unit tests for the f1data data-engineering package (Phase 1, Task 4).

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from app.intelligence.data import eda, synthetic
from app.intelligence.data.pipeline import (
    clean_table,
    encode_and_scale,
    load_csv,
    parse_laptime_to_seconds,
)
from app.intelligence.data.schemas import ALL_SCHEMAS, ColKind


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """Generate a synthetic dataset once for the whole test module."""
    d = tmp_path_factory.mktemp("raw")
    synthetic.generate_all(d, seed=123)
    return d


# --------------------------------------------------------------------------- #
# Lap-time parsing
# --------------------------------------------------------------------------- #

def test_parse_laptime_minutes_seconds():
    assert parse_laptime_to_seconds("1:30.500") == pytest.approx(90.5)
    assert parse_laptime_to_seconds("0:59.999") == pytest.approx(59.999)


def test_parse_laptime_plain_seconds():
    assert parse_laptime_to_seconds("88.421") == pytest.approx(88.421)


def test_parse_laptime_handles_nulls():
    assert np.isnan(parse_laptime_to_seconds(r"\N"))
    assert np.isnan(parse_laptime_to_seconds(""))
    assert np.isnan(parse_laptime_to_seconds(None))
    assert np.isnan(parse_laptime_to_seconds("not-a-time"))


# --------------------------------------------------------------------------- #
# Synthetic generation
# --------------------------------------------------------------------------- #

def test_generate_all_creates_expected_files(dataset):
    for name in ("races", "drivers", "constructors", "circuits", "results",
                 "pit_stops", "lap_times", "fastf1_laps"):
        assert (dataset / f"{name}.csv").exists(), name


def test_generated_files_match_schema_columns(dataset):
    for name, schema in ALL_SCHEMAS.items():
        df = pd.read_csv(dataset / f"{name}.csv")
        assert set(schema.column_names()) <= set(df.columns), name


def test_synthetic_injects_issues(dataset):
    # results.csv should contain injected duplicate rows / NaNs somewhere.
    results = pd.read_csv(dataset / "results.csv")
    assert results.duplicated().any() or results.isna().any().any()


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #

def test_load_converts_null_sentinels(tmp_path):
    p = tmp_path / "drivers.csv"
    p.write_text("driverId,driverRef,code,forename,surname,nationality\n"
                 "1,x,XXX,A,B,\\N\n")
    df = load_csv(p, ALL_SCHEMAS["drivers"])
    assert df["nationality"].isna().iloc[0]


def test_load_rejects_missing_primary_key(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        load_csv(p, ALL_SCHEMAS["results"])


def test_clean_removes_duplicates_and_imputes(dataset):
    schema = ALL_SCHEMAS["results"]
    df = load_csv(dataset / "results.csv", schema)
    before = len(df)
    cdf, report = clean_table(df, schema, cap_outliers=True)
    # No duplicate rows remain.
    assert not cdf.duplicated().any()
    # Numeric columns fully imputed (no NaN in points).
    assert not cdf["points"].isna().any()
    # The audit records a deduplicate step.
    assert any(s.step == "deduplicate" for s in report.steps)
    assert len(cdf) <= before


def test_clean_parses_time_columns_to_float(dataset):
    schema = ALL_SCHEMAS["lap_times"]
    df = load_csv(dataset / "lap_times.csv", schema)
    cdf, _ = clean_table(df, schema)
    # 'time' is a TIME column and must become float seconds.
    assert pd.api.types.is_float_dtype(cdf["time"])
    assert cdf["time"].dropna().gt(0).all()


def test_clean_caps_outliers(dataset):
    schema = ALL_SCHEMAS["lap_times"]
    df = load_csv(dataset / "lap_times.csv", schema)
    capped, report = clean_table(df, schema, cap_outliers=True)
    uncapped, _ = clean_table(df, schema, cap_outliers=False)
    ms = "milliseconds"
    assert capped[ms].max() <= uncapped[ms].max()
    assert any(s.step == "detect_outliers" for s in report.steps)


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #

def _clean(dataset, name):
    schema = ALL_SCHEMAS[name]
    df = load_csv(dataset / f"{name}.csv", schema)
    cdf, _ = clean_table(df, schema, cap_outliers=True)
    return cdf


def test_statistical_summary_has_moments(dataset):
    s = eda.statistical_summary(_clean(dataset, "lap_times"))
    assert {"skew", "kurtosis", "missing"} <= set(s.columns)


def test_data_quality_scorecard(dataset):
    q = eda.data_quality(_clean(dataset, "results"), "results")
    assert 0 <= q.quality_score <= 100
    assert q.completeness <= 1.0
    assert q.duplicate_rows == 0  # cleaning removed them


def test_correlation_returns_pairs(dataset):
    result = eda.correlation_analysis(_clean(dataset, "fastf1_laps"))
    assert not result.matrix.empty
    assert result.top_pairs  # at least one correlated pair
    # Sector times should correlate strongly with lap time.
    keys = {tuple(sorted((a, b))) for a, b, _ in result.top_pairs}
    assert any("LapTime" in pair for pair in keys)


def test_driver_analysis_ranks_by_points(dataset):
    d = eda.driver_analysis(_clean(dataset, "results"),
                            _clean(dataset, "drivers"))
    assert "total_points" in d.columns
    # Sorted descending.
    assert d["total_points"].is_monotonic_decreasing


def test_tyre_analysis_orders_by_pace(dataset):
    t = eda.tyre_analysis(_clean(dataset, "fastf1_laps"))
    assert {"compound", "mean_laptime_s", "deg_s_per_lap"} <= set(t.columns)
    # SOFT should degrade faster than HARD (physics baked into the generator).
    deg = t.set_index("compound")["deg_s_per_lap"]
    if "SOFT" in deg and "HARD" in deg:
        assert deg["SOFT"] >= deg["HARD"]


# --------------------------------------------------------------------------- #
# Encoding & scaling
# --------------------------------------------------------------------------- #

def test_encode_and_scale_transforms_features(dataset):
    schema = ALL_SCHEMAS["fastf1_laps"]
    cdf = _clean(dataset, "fastf1_laps")
    result = encode_and_scale(cdf, schema)
    # Low-cardinality categoricals one-hot expanded -> more columns.
    assert result.frame.shape[1] >= cdf.shape[1]
    # Scaled numeric columns have ~zero mean.
    for col in result.scaled_columns:
        assert abs(result.frame[col].astype(float).mean()) < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
