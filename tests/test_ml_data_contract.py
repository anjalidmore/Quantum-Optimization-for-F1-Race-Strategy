"""Task 6 data-contract tests: the Task 5 -> Task 6 hand-off must hold."""
from __future__ import annotations

import pandas as pd
import pytest

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
