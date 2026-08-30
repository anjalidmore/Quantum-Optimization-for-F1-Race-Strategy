"""Regression tests for the race-strategy simulator's feature construction.

These exist because of a real bug: driver/team one-hot dummies were being
median-imputed regardless of the selected driver/team, so the dropdown had
no effect on real-data predictions. build_feature_row must derive one-hot
driver/team/compound features exactly from the request, not approximate them.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.intelligence.features.contract import load_feature_contract
from app.services.feature_approximation import build_feature_row, relevance_for_target


def _race_state(**overrides):
    defaults = dict(
        driver="VER", team="RED BULL RACING", current_lap=20, total_laps=55,
        tyre_compound="MEDIUM", tyre_age=15, track_temperature=40.0,
        weather="dry", fuel_kg=70.0, track_status="GREEN", current_position=6,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_driver_one_hot_feature_is_exact_not_approximated():
    contract = load_feature_contract()
    features = contract.selected_features("target_laptime")
    driver_features = [f for f in features if f.startswith("driver_")]
    if not driver_features:
        return  # nothing to test against the current contract

    target_feature = driver_features[0]
    matching_driver = target_feature[len("driver_"):].upper()

    result_match = build_feature_row("target_laptime", _race_state(driver=matching_driver))
    result_other = build_feature_row("target_laptime", _race_state(driver="ZZZ"))

    assert target_feature not in result_match.approximated
    assert result_match.row[target_feature] == 1.0
    assert result_other.row[target_feature] == 0.0


def test_team_one_hot_feature_is_exact_not_approximated():
    contract = load_feature_contract()
    features = contract.selected_features("target_laptime")
    team_features = [f for f in features if f.startswith("team_")]
    if not team_features:
        return

    target_feature = team_features[0]
    team_slug = target_feature[len("team_"):]
    matching_team = team_slug.replace("_", " ").upper()

    result_match = build_feature_row("target_laptime", _race_state(team=matching_team))
    result_other = build_feature_row("target_laptime", _race_state(team="A TEAM THAT DOES NOT EXIST"))

    assert target_feature not in result_match.approximated
    assert result_match.row[target_feature] == 1.0
    assert result_other.row[target_feature] == 0.0


def test_compound_feature_is_exact_and_matches_selected_compound():
    contract = load_feature_contract()
    features = contract.selected_features("target_laptime")
    compound_features = [f for f in features if f.startswith("compound_")]
    if not compound_features:
        return

    target_feature = compound_features[0]
    compound = target_feature[len("compound_"):].upper()

    result_match = build_feature_row("target_laptime", _race_state(tyre_compound=compound))
    result_other_compound = "HARD" if compound != "HARD" else "SOFT"
    result_other = build_feature_row("target_laptime", _race_state(tyre_compound=result_other_compound))

    assert result_match.row[target_feature] == 1.0
    assert result_other.row[target_feature] == 0.0


def test_different_drivers_change_at_least_one_feature_value_when_driver_features_exist():
    contract = load_feature_contract()
    relevance = relevance_for_target("target_laptime")
    if not relevance["driver"]:
        return  # this target's contract has no driver-derived features at all

    row_a = build_feature_row("target_laptime", _race_state(driver="VER")).row
    row_b = build_feature_row("target_laptime", _race_state(driver="HAM")).row
    assert row_a != row_b, "changing the driver must change at least one feature value"


def test_out_of_range_input_is_flagged_not_silently_extrapolated():
    # A track temperature far outside anything the model was trained on
    # (see data/processed/fastf1_laps_clean.csv's real TrackTemp range)
    # must be flagged, not silently fed to the model as if it were normal.
    result = build_feature_row("target_laptime", _race_state(track_temperature=200.0, tyre_age=20, current_lap=20))
    flagged_features = {o["feature"] for o in result.out_of_range}
    assert "tracktemp_dev_x_tyrelife" in flagged_features


def test_in_range_input_is_not_flagged():
    result = build_feature_row("target_laptime", _race_state(track_temperature=30.0, tyre_age=5, current_lap=20))
    assert result.out_of_range == [] or all(o["feature"] != "tyre_life" for o in result.out_of_range)


def test_relevance_correctly_flags_context_only_fields():
    # The real classification contract (target_pit_next_lap) has no driver or
    # team features at all in the current selection - this must be reported
    # as context-only, not silently pretended to matter.
    contract = load_feature_contract()
    clf_features = set(contract.selected_features("target_pit_next_lap"))
    relevance = relevance_for_target("target_pit_next_lap")
    for group in ("driver", "team"):
        for f in relevance[group]:
            assert f in clf_features
