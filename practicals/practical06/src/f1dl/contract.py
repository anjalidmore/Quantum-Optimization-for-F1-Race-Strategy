"""
f1dl.contract
=============

Reads the Task 5 feature contract produced by practical05 and gates every
guarantee Task 7 depends on before a single network is trained.

This mirrors Task 6's ``data_contract`` module: Task 7 deliberately does not
engineer any new features. It consumes exactly the causal, leakage-checked
feature set Task 5 exported, so that a deep network and a classical model are
compared on identical inputs and any difference in score is attributable to
the model rather than to the data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DataContractError(RuntimeError):
    """Raised when the Task 5 contract is violated. Task 7 fails loudly
    rather than training on a silently broken contract."""


def find_practicals_root(start: Path) -> Path:
    """Walk upwards until the folder containing the sibling practicals is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "practical05").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate 'practical05'. Run this practical from inside the project tree."
    )


@dataclass(frozen=True)
class FeatureContract:
    metadata: dict

    @property
    def identifier_columns(self) -> list[str]:
        return list(self.metadata["identifier_columns"])

    @property
    def target_names(self) -> list[str]:
        return list(self.metadata["targets"].keys())

    @property
    def union_exported_features(self) -> list[str]:
        return list(self.metadata["selected_features"]["union_exported"])

    @property
    def numeric_features_requiring_scaling(self) -> list[str]:
        return list(self.metadata["numeric_features_requiring_scaling"])

    @property
    def binary_features_no_scaling_needed(self) -> list[str]:
        return list(self.metadata["binary_features_no_scaling_needed"])

    @property
    def leakage_columns(self) -> list[str]:
        return list(self.metadata["excluded_as_leakage"]["columns"])

    @property
    def leakage_reason(self) -> str:
        return str(self.metadata["excluded_as_leakage"]["reason"])

    @property
    def source_dataset(self) -> str:
        return str(self.metadata.get("source_dataset", "unknown"))

    def selected_features(self, target: str) -> list[str]:
        return list(self.metadata["selected_features"][target])

    def identity_features(self, target: str) -> list[str]:
        """Driver/team one-hot dummies among a target's selected features.

        Task 8's fairness assessment keys off this: these encode *who* is
        driving rather than *what the car is doing*, so a model leaning on
        them is fitting identity, not race state.
        """
        return [f for f in self.selected_features(target) if f.startswith(("driver_", "team_"))]


@dataclass(frozen=True)
class DLDataset:
    frame: pd.DataFrame
    contract: FeatureContract


def load_and_validate(practical05_outputs: Path | None = None) -> DLDataset:
    if practical05_outputs is None:
        root = find_practicals_root(Path.cwd())
        practical05_outputs = root / "practical05" / "outputs"

    csv = practical05_outputs / "f1_features_selected.csv"
    meta = practical05_outputs / "feature_metadata.json"
    for p in (csv, meta):
        if not p.exists():
            raise DataContractError(
                f"Task 5 contract file missing: {p}. Run practical05 first "
                f"(see practicals/practical05/practical05.ipynb)."
            )

    frame = pd.read_csv(csv)
    contract = FeatureContract(metadata=json.loads(meta.read_text()))

    _validate_columns_present(frame, contract)
    _validate_no_leakage(frame, contract)
    _validate_no_missing_values(frame, contract)
    return DLDataset(frame=frame, contract=contract)


def _validate_columns_present(frame: pd.DataFrame, contract: FeatureContract) -> None:
    missing = [c for c in contract.identifier_columns if c not in frame.columns]
    if missing:
        raise DataContractError(f"Identifier columns missing from feature matrix: {missing}")
    missing = [c for c in contract.target_names if c not in frame.columns]
    if missing:
        raise DataContractError(f"Target columns missing from feature matrix: {missing}")
    for target in ("target_laptime", "target_pit_next_lap"):
        missing = [c for c in contract.selected_features(target) if c not in frame.columns]
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
        leaking = set(contract.selected_features(target)) & all_targets
        if leaking:
            raise DataContractError(
                f"Target column(s) {leaking} appear in the feature list for "
                f"{target!r} - a target cannot be its own predictor."
            )


def _validate_no_missing_values(frame: pd.DataFrame, contract: FeatureContract) -> None:
    cols = (set(contract.union_exported_features) | set(contract.target_names)) & set(frame.columns)
    na = frame[list(cols)].isna().sum()
    bad = na[na > 0]
    if not bad.empty:
        raise DataContractError(f"Unexpected missing values in feature matrix: {bad.to_dict()}")


def build_task_frame(
    dataset: DLDataset, target: str, lap_col: str = "LapNumber"
) -> tuple[pd.DataFrame, list[str]]:
    """A freshly-indexed (0..N-1) frame of exactly this target's features,
    the target, and the lap column, so fold indices stay aligned by
    construction."""
    features = dataset.contract.selected_features(target)
    frame = dataset.frame[features + [target, lap_col]].reset_index(drop=True)
    return frame, features
