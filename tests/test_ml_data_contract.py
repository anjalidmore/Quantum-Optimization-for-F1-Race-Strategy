"""Task 6 data-contract tests: the Task 5 -> Task 6 hand-off must hold.

Also covers the Task 5 builder itself (``app.intelligence.features.build``), which
produces that contract: its leakage exclusions and its lap-forward splits are the
two rules everything downstream depends on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.intelligence.features import build as feature_build
from app.intelligence.features.contract import load_feature_contract, load_feature_matrix
from app.intelligence.ml.data_contract import DataContractError, build_task_frame, load_and_validate


def test_feature_metadata_and_matrix_exist():
    contract = load_feature_contract()
    matrix = load_feature_matrix()
    assert contract.raw
    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) > 0


def test_selected_features_are_columns_in_matrix():
    contract = load_feature_contract()
    matrix = load_feature_matrix()
    for target in ("target_laptime", "target_pit_next_lap"):
        for feature in contract.selected_features(target):
            assert feature in matrix.columns


def test_targets_present():
    contract = load_feature_contract()
    matrix = load_feature_matrix()
    for target in contract.target_names:
        assert target in matrix.columns


def test_leakage_columns_absent_from_matrix():
    contract = load_feature_contract()
    matrix = load_feature_matrix()
    for col in contract.leakage_columns:
        assert col not in matrix.columns, f"Leakage column {col!r} must never enter the training matrix"


def test_leakage_columns_absent_from_every_selected_feature_list():
    contract = load_feature_contract()
    for target in ("target_laptime", "target_pit_next_lap"):
        selected = set(contract.selected_features(target))
        assert not selected & set(contract.leakage_columns)


def test_target_not_used_as_its_own_predictor():
    contract = load_feature_contract()
    for target in ("target_laptime", "target_pit_next_lap"):
        assert target not in contract.selected_features(target)


def test_load_and_validate_succeeds_on_real_contract():
    dataset = load_and_validate()
    assert len(dataset.frame) > 0


def test_build_task_frame_contains_exactly_selected_features_plus_target_and_lap():
    dataset = load_and_validate()
    frame, features = build_task_frame(dataset, "target_laptime")
    assert set(frame.columns) == set(features) | {"target_laptime", "LapNumber"}
    assert list(frame.index) == list(range(len(frame)))


def test_data_contract_error_on_missing_leakage_free_guarantee():
    dataset = load_and_validate()
    corrupted = dataset.frame.copy()
    corrupted["Sector1Time"] = 0.0
    from app.intelligence.ml.data_contract import _validate_no_leakage

    with pytest.raises(DataContractError):
        _validate_no_leakage(corrupted, dataset.contract)


# ---------------------------------------------------------------------------
# The Task 5 builder (ported out of the notebook on 2026-09-27)
# ---------------------------------------------------------------------------
def test_committed_contract_matches_what_the_builder_would_select():
    """The committed metadata must name the same feature counts the builder uses.

    A cheap guard that the contract on disk came from this code and not from a
    stale notebook run. The full value-for-value check is
    `python scripts/build_features.py --check`.
    """
    contract = load_feature_contract()
    assert len(contract.selected_features("target_pit_next_lap")) == feature_build.K_CLASSIFICATION
    assert contract.raw["random_state"] == feature_build.RANDOM_STATE
    funnel = contract.raw["selection_funnel"]["4_importance_and_stability"]
    assert "within 2% of best CV MAE" in funnel


def test_builder_excludes_every_leakage_column():
    matrix = load_feature_matrix()
    for column in feature_build.LEAKAGE_COLS:
        assert column not in matrix.columns, f"{column} is leakage and must never be exported"


def test_lap_forward_splits_never_train_on_a_later_lap():
    laps = pd.Series(np.repeat(np.arange(1, 21), 3))      # 20 laps, 3 drivers
    splits = list(feature_build.lap_forward_splits(laps))
    assert splits
    for train_idx, test_idx in splits:
        assert laps.iloc[train_idx].max() < laps.iloc[test_idx].min()
        # whole laps stay on one side, or a field-median feature would leak across
        assert set(laps.iloc[train_idx]).isdisjoint(set(laps.iloc[test_idx]))


def test_near_zero_variance_drops_a_constant_column():
    frame = pd.DataFrame({"useful": np.arange(100.0), "constant": np.ones(100)})
    assert feature_build.near_zero_variance(frame) == ["constant"]
