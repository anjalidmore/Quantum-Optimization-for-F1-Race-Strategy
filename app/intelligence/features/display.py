"""
app.intelligence.features.display
====================================

Human-readable names/descriptions for Task 5 feature columns, for the
frontend — a professor (or anyone unfamiliar with the internal feature
matrix) should never have to read a raw column name like
``tracktemp_dev_x_tyrelife`` or ``driver_ver`` unexplained.

Curated overrides cover the hand-engineered features from
``docs/notebooks/task5_feature_engineering.ipynb``; anything else (in
particular, the one-hot driver/team/compound dummies, whose exact set
depends on which drivers/teams were in whatever session was trained on) is
named generically from the column's own naming convention plus the
Task 5-generated ``feature_provenance`` description, so this never goes
stale when the dataset changes.
"""
from __future__ import annotations

import re

from app.intelligence.features.contract import FeatureContract

_OVERRIDE_NAMES: dict[str, str] = {
    "tyre_life": "Tyre Age",
    "gap_roll3_mean": "Recent Pace Gap",
    "gap_roll3_std": "Pace Gap Volatility",
    "gap_expanding": "Overall Pace Gap Trend",
    "field_median_lag1": "Previous-Lap Field Pace",
    "field_pace_trend": "Field Pace Trend",
    "form_vs_baseline": "Current Form vs Baseline",
    "race_progress": "Race Progress",
    "stint_number": "Stint Number",
    "track_status": "Track Status Flag",
    "wind_speed": "Wind Speed",
    "humidity": "Relative Humidity",
    "is_fresh_tyre": "Fresh Tyre Indicator",
    "compound_soft": "Soft Compound Indicator",
    "compound_medium": "Medium Compound Indicator",
    "tyrelife_x_soft": "Tyre Age × Soft Compound",
    "tyrelife_x_medium": "Tyre Age × Medium Compound",
    "tracktemp_dev_x_tyrelife": "Track Temperature Deviation × Tyre Age",
    "team_aston_martin": "Aston Martin Team Indicator",
    "driver_sai": "Driver Performance Index",  # only used when NOT a one-hot dummy — see display_name()
}

_UNIT_HINTS: dict[str, str] = {
    "tyre_life": "laps",
    "race_progress": "fraction (0-1)",
    "wind_speed": "m/s",
    "humidity": "%",
    "track_temperature": "°C",
}

_DRIVER_RE = re.compile(r"^driver_([a-z0-9]+)$")
_TEAM_RE = re.compile(r"^team_([a-z0-9_]+)$")
_COMPOUND_RE = re.compile(r"^compound_([a-z]+)$")


def display_name(feature: str, contract: FeatureContract) -> str:
    is_one_hot = feature in set(contract.binary_features_no_scaling_needed)

    m = _DRIVER_RE.match(feature)
    if m and is_one_hot:
        return f"Driver: {m.group(1).upper()}"
    m = _TEAM_RE.match(feature)
    if m and is_one_hot:
        return f"Team: {m.group(1).replace('_', ' ').title()}"
    m = _COMPOUND_RE.match(feature)
    if m:
        return f"Compound: {m.group(1).title()}"

    if feature in _OVERRIDE_NAMES:
        return _OVERRIDE_NAMES[feature]

    return feature.replace("_", " ").title()


def description(feature: str, contract: FeatureContract) -> str:
    return contract.raw.get("feature_provenance", {}).get(feature, "No description recorded for this feature.")


def unit_hint(feature: str) -> str | None:
    return _UNIT_HINTS.get(feature)


def describe_features(features: list[str], contract: FeatureContract, stats: dict | None = None) -> list[dict]:
    """``stats``: optional {feature: {"min":..., "median":..., "max":...}}
    computed from real training data (see ``load_feature_matrix``), so the
    frontend can offer sensible bounds/defaults instead of arbitrary ones."""
    stats = stats or {}
    out = []
    for f in features:
        entry = {
            "feature": f,
            "display_name": display_name(f, contract),
            "description": description(f, contract),
            "unit": unit_hint(f),
        }
        if f in stats:
            entry.update(stats[f])
        out.append(entry)
    return out


def feature_stats(features: list[str], frame) -> dict[str, dict]:
    """Real min/median/max per feature from the training data — used as
    sensible input bounds/defaults, never arbitrary numbers."""
    out = {}
    for f in features:
        if f in frame.columns:
            col = frame[f]
            out[f] = {"min": float(col.min()), "median": float(col.median()), "max": float(col.max())}
    return out
