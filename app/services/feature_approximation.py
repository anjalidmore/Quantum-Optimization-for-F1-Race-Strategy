"""
app.services.feature_approximation
=====================================

The Task 6 ML models require Task 5's engineered features. The race-strategy
simulator instead receives a single race-state *snapshot* (driver, team,
tyre compound, tyre age, lap, ...) from a form. This module maps that
snapshot onto a feature row — and, critically, does so **generically from
the live feature contract**, not from a fixed list of feature names.

Why generic: Task 5's feature-selection notebook one-hot-encodes driver and
team identity as `driver_<code>` / `team_<slug>` columns (see
``docs/notebooks/task5_feature_engineering.ipynb``, "Block F"), and which of
those columns survive selection depends on the dataset actually trained on
(the synthetic demo selects a `driver_sai` *skill-index* feature; a real
FastF1 session instead selects one-hot dummies for whichever drivers/teams
were in that race, and coincidentally one of Task 5's real-data dummies is
also literally named `driver_sai` because "SAI" is Carlos Sainz's code).
Hard-coding "driver_sai means the skill index" broke silently on real data:
every one-hot driver/team dummy was falling through to the median-fallback
branch, meaning the selected driver/team had **no effect at all** on
predictions. This module instead classifies each selected feature at
request time by matching it against the driver/team/compound the user
actually picked, using the contract's own binary-vs-numeric classification
to tell a one-hot dummy apart from a continuous per-driver statistic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from app.core.paths import FASTF1_LAPS_CLEAN_CSV
from app.intelligence.features.contract import load_feature_contract, load_feature_matrix

_DRIVER_RE = re.compile(r"^driver_([a-z0-9]+)$")
_TEAM_RE = re.compile(r"^team_([a-z0-9_]+)$")
_COMPOUND_ONEHOT_RE = re.compile(r"^compound_([a-z]+)$")
_TYRELIFE_INTERACTION_RE = re.compile(r"^tyrelife_x_([a-z]+)$")


@dataclass
class FeatureRowResult:
    row: dict[str, float]
    # Feature names whose value was derived exactly from a user-provided
    # field (driver/team/compound/tyre_age/lap), grouped by that field.
    derived_from: dict[str, list[str]] = field(default_factory=lambda: {"driver": [], "team": [], "compound": [], "race_state": []})
    # Feature names that could not be derived from the snapshot and fell
    # back to a training-data median (history-dependent features).
    approximated: list[str] = field(default_factory=list)
    # Feature/value pairs that fall outside the range the model was ever
    # trained on, with the training range for context. A prediction built on
    # these is extrapolation, not interpolation — the model's behaviour out
    # there is unvalidated and can be arbitrary (e.g. a shallow decision
    # tree has no leaf that was ever tested against such an input). This is
    # computed from the actual training data's min/max, never guessed.
    out_of_range: list[dict] = field(default_factory=list)


@lru_cache(maxsize=1)
def _training_frame() -> pd.DataFrame:
    return load_feature_matrix()


@lru_cache(maxsize=1)
def _mean_track_temp() -> float:
    """The Task 4 clean dataset carries raw TrackTemp; Task 5's interaction
    term (`tracktemp_dev_x_tyrelife`) mean-centres it. Reuse the same
    dataset to reproduce that centring constant exactly, rather than
    guessing it."""
    if FASTF1_LAPS_CLEAN_CSV.exists():
        df = pd.read_csv(FASTF1_LAPS_CLEAN_CSV)
        if "TrackTemp" in df.columns:
            return float(df["TrackTemp"].mean())
    return 35.0  # falls back to RaceProblem's own default track temp


def relevance_for_target(target: str) -> dict[str, list[str]]:
    """Which selected features for ``target`` are influenced by driver,
    team, or tyre compound — used by the API/frontend to honestly say
    "context only, not used by this model" when a field has zero effect on
    the currently trained model."""
    contract = load_feature_contract()
    features = contract.selected_features(target)
    relevant: dict[str, list[str]] = {"driver": [], "team": [], "compound": []}
    for f in features:
        if _DRIVER_RE.match(f) or f == "driver_sai":
            relevant["driver"].append(f)
        elif _TEAM_RE.match(f):
            relevant["team"].append(f)
        elif _COMPOUND_ONEHOT_RE.match(f) or _TYRELIFE_INTERACTION_RE.match(f):
            relevant["compound"].append(f)
    return relevant


def build_feature_row(target: str, race_state) -> FeatureRowResult:
    contract = load_feature_contract()
    features = contract.selected_features(target)
    df = _training_frame()
    binary_set = set(contract.binary_features_no_scaling_needed)

    driver_code = race_state.driver.strip().upper()
    driver_slug = driver_code.lower()
    team_name = race_state.team.strip()
    team_slug = team_name.lower().replace(" ", "_")
    compound = race_state.tyre_compound.strip().upper()
    compound_slug = compound.lower()

    race_progress = (race_state.current_lap / race_state.total_laps) if race_state.total_laps else 0.0
    track_temp_dev = race_state.track_temperature - _mean_track_temp()

    result = FeatureRowResult(row={})

    for feature in features:
        if feature == "tyre_life":
            result.row[feature] = float(race_state.tyre_age)
            result.derived_from["race_state"].append(feature)
            continue
        if feature == "race_progress":
            result.row[feature] = float(race_progress)
            result.derived_from["race_state"].append(feature)
            continue
        if feature == "tracktemp_dev_x_tyrelife":
            result.row[feature] = float(track_temp_dev * race_state.tyre_age)
            result.derived_from["race_state"].append(feature)
            continue

        m = _TYRELIFE_INTERACTION_RE.match(feature)
        if m:
            result.row[feature] = float(race_state.tyre_age) if compound_slug == m.group(1) else 0.0
            result.derived_from["compound"].append(feature)
            continue

        m = _COMPOUND_ONEHOT_RE.match(feature)
        if m:
            result.row[feature] = 1.0 if compound_slug == m.group(1) else 0.0
            result.derived_from["compound"].append(feature)
            continue

        m = _DRIVER_RE.match(feature) or (feature == "driver_sai" and re.match(r"^driver_([a-z0-9]+)$", feature))
        if m:
            code = m.group(1) if hasattr(m, "group") else feature[len("driver_"):]
            if feature in binary_set:
                # A genuine one-hot driver-identity dummy: exactly 1 for the
                # matching driver, 0 otherwise (including the reference
                # driver, who has no dummy column at all).
                result.row[feature] = 1.0 if driver_slug == code else 0.0
            else:
                # A continuous per-driver statistic (e.g. the synthetic
                # demo's driver-skill index) — use that driver's own median
                # from training data, falling back to the overall median if
                # the driver never appears in it.
                rows = df[df["Driver"].str.upper() == driver_code] if "Driver" in df.columns else df.iloc[0:0]
                result.row[feature] = float(rows[feature].median()) if len(rows) else float(df[feature].median())
            result.derived_from["driver"].append(feature)
            continue

        m = _TEAM_RE.match(feature)
        if m:
            slug = m.group(1)
            if feature in binary_set:
                result.row[feature] = 1.0 if team_slug == slug else 0.0
            else:
                rows = df[df["Team"].str.upper() == team_name.upper()] if "Team" in df.columns else df.iloc[0:0]
                result.row[feature] = float(rows[feature].median()) if len(rows) else float(df[feature].median())
            result.derived_from["team"].append(feature)
            continue

        if feature == "team_aston_martin":
            # Legacy synthetic-contract feature name, kept for backward
            # compatibility with old registries: a plain equality flag, not
            # a one-hot dummy derived from the live team list.
            result.row[feature] = 1.0 if team_name.upper() == "ASTON MARTIN" else 0.0
            result.derived_from["team"].append(feature)
            continue

        # Everything else requires multi-lap race history the snapshot
        # can't supply (rolling gaps, field pace, form vs. baseline, ...).
        result.row[feature] = float(df[feature].median())
        result.approximated.append(feature)

    _flag_out_of_range(result, df, binary_set)
    return result


def _flag_out_of_range(result: FeatureRowResult, df: pd.DataFrame, binary_set: set[str]) -> None:
    """Flag any exactly-derived feature value that falls outside the range
    the model was actually trained on. One-hot (binary) features are always
    0/1 by construction and skipped; approximated (median-filled) features
    are trivially in range and skipped. A flagged feature means the
    prediction is extrapolating beyond validated model behaviour — most
    often because a form/preset input (e.g. track temperature) doesn't
    match the real conditions of the session the model was trained on."""
    for feature, value in result.row.items():
        if feature in result.approximated or feature in binary_set or feature not in df.columns:
            continue
        col_min, col_max = float(df[feature].min()), float(df[feature].max())
        if value < col_min or value > col_max:
            result.out_of_range.append(
                {"feature": feature, "value": value, "training_min": col_min, "training_max": col_max}
            )
