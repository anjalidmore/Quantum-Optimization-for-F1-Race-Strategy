"""
app.intelligence.ml.data_contract
====================================

Task 6's data-contract gate. Loads the Task 5 outputs through
``app.intelligence.features.contract`` and validates every guarantee Task 6
depends on before a single model is trained:

* the identifier columns, targets, and selected feature lists the metadata
  promises actually exist as columns in the CSV
* none of the columns Task 5 documented as leakage
  (``excluded_as_leakage.columns``) are present in the feature matrix at all
* no target column leaks into another target's feature list
* the feature matrix has no missing values in the columns Task 6 will use

``validate()`` raises ``DataContractError`` on any violation — Task 6 must
fail loudly rather than train on a silently broken contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.intelligence.features.contract import (
    FeatureContract,
    load_feature_contract,
    load_feature_matrix,
)


class DataContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class MLDataset:
    frame: pd.DataFrame
    contract: FeatureContract


def load_and_validate() -> MLDataset:
    contract = load_feature_contract()
    frame = load_feature_matrix()

    _validate_columns_present(frame, contract)
    _validate_no_leakage(frame, contract)
    _validate_no_missing_values(frame, contract)

    return MLDataset(frame=frame, contract=contract)


def _validate_columns_present(frame: pd.DataFrame, contract: FeatureContract) -> None:
    missing = [c for c in contract.identifier_columns if c not in frame.columns]
    if missing:
        raise DataContractError(f"Identifier columns missing from feature matrix: {missing}")

    missing = [c for c in contract.target_names if c not in frame.columns]
    if missing:
        raise DataContractError(f"Target columns missing from feature matrix: {missing}")

    for target in ("target_laptime", "target_pit_next_lap"):
        selected = contract.selected_features(target)
        missing = [c for c in selected if c not in frame.columns]
        if missing:
            raise DataContractError(
                f"Selected features for {target!r} missing from feature matrix: {missing}"
            )


def _validate_no_leakage(frame: pd.DataFrame, contract: FeatureContract) -> None:
    present = [c for c in contract.leakage_columns if c in frame.columns]
    if present:
        raise DataContractError(
            f"Leakage columns present in the training matrix: {present}. "
            f"Reason these are excluded: {contract.leakage_reason}"
        )

    all_targets = set(contract.target_names)
    for target in ("target_laptime", "target_pit_next_lap"):
        selected = set(contract.selected_features(target))
        leaking_targets = selected & all_targets
        if leaking_targets:
            raise DataContractError(
                f"Target column(s) {leaking_targets} appear in the feature "
                f"list for {target!r} — a target cannot be its own predictor."
            )


def _validate_no_missing_values(frame: pd.DataFrame, contract: FeatureContract) -> None:
    cols = set(contract.union_exported_features) | set(contract.target_names)
    cols &= set(frame.columns)
    na_counts = frame[list(cols)].isna().sum()
    bad = na_counts[na_counts > 0]
    if not bad.empty:
        raise DataContractError(f"Unexpected missing values in feature matrix: {bad.to_dict()}")


def build_task_frame(dataset: MLDataset, target: str, lap_col: str = "LapNumber") -> tuple[pd.DataFrame, list[str]]:
    """A single, freshly-indexed (0..N-1) DataFrame holding exactly the
    selected features for ``target``, the target itself, and ``lap_col``.

    Splitting (``ml.splits``) and tuning (``ml.tuning``) both key off
    positional indices derived from *this* frame's index, so everything
    downstream — X, y, and the lap-based fold boundaries — stays aligned by
    construction rather than by convention.
    """
    features = dataset.contract.selected_features(target)
    columns = features + [target, lap_col]
    frame = dataset.frame[columns].reset_index(drop=True)
    return frame, features


def get_xy(frame: pd.DataFrame, target: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Split a task frame (as built by ``build_task_frame``) into X, y,
    preserving its index so positional fold indices remain valid."""
    return frame[features], frame[target]
