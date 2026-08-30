"""
app.intelligence.features.contract
====================================

Reads the Task 5 feature-engineering outputs and exposes them as a typed,
machine-readable contract:

    data/processed/f1_features_selected.csv   -> the model-ready feature matrix
    data/processed/feature_metadata.json      -> which columns mean what,
                                                  which are selected per task,
                                                  which are leakage, and the
                                                  validation strategy to use.

Nothing here recomputes or alters a single feature value. This is a thin,
strict reader so that Task 6 (and everything after it) treats Task 5's
output as an immutable contract rather than a file it happens to load.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.core.paths import TASK5_FEATURE_METADATA_JSON, TASK5_FEATURES_CSV


class FeatureContractError(RuntimeError):
    """Raised when the Task 5 outputs are missing or violate the contract."""


@dataclass(frozen=True)
class FeatureContract:
    """Typed view over ``feature_metadata.json``."""

    raw: dict = field(repr=False)

    @property
    def identifier_columns(self) -> list[str]:
        return list(self.raw["identifier_columns"])

    @property
    def targets(self) -> dict[str, str]:
        return dict(self.raw["targets"])

    @property
    def target_names(self) -> list[str]:
        return list(self.raw["targets"].keys())

    def selected_features(self, target: str) -> list[str]:
        try:
            return list(self.raw["selected_features"][target])
        except KeyError as exc:
            raise FeatureContractError(
                f"No selected-feature list for target={target!r} in "
                f"feature_metadata.json. Known targets: "
                f"{list(self.raw['selected_features'])}"
            ) from exc

    @property
    def union_exported_features(self) -> list[str]:
        return list(self.raw["selected_features"]["union_exported"])

    @property
    def leakage_columns(self) -> list[str]:
        return list(self.raw["excluded_as_leakage"]["columns"])

    @property
    def leakage_reason(self) -> str:
        return str(self.raw["excluded_as_leakage"]["reason"])

    @property
    def numeric_features_requiring_scaling(self) -> list[str]:
        return list(self.raw.get("numeric_features_requiring_scaling", []))

    @property
    def binary_features_no_scaling_needed(self) -> list[str]:
        return list(self.raw.get("binary_features_no_scaling_needed", []))

    @property
    def validation_strategy(self) -> str:
        return str(self.raw["preprocessing_contract"]["validation"])

    @property
    def scaling_policy(self) -> str:
        return str(self.raw["preprocessing_contract"]["scaling"])

    @property
    def synthetic_caveat(self) -> str | None:
        return self.raw.get("validation_scores", {}).get("caveat")

    @property
    def random_state(self) -> int:
        return int(self.raw.get("random_state", 42))

    @property
    def source_dataset(self) -> str:
        return str(self.raw.get("source_dataset", ""))

    @property
    def dataset_source(self) -> dict:
        """Real-vs-synthetic provenance, propagated from
        scripts/fetch_real_session.py through scripts/run_eda.py. Defaults to
        synthetic only when the notebook genuinely found no marker — never
        assumed at the reporting/API/frontend layer."""
        return dict(self.raw.get("dataset_source", {"source": "synthetic"}))

    @property
    def is_real_data(self) -> bool:
        return self.dataset_source.get("source") == "real_fastf1"


@lru_cache(maxsize=1)
def load_feature_contract(path: Path = TASK5_FEATURE_METADATA_JSON) -> FeatureContract:
    if not path.exists():
        raise FeatureContractError(
            f"Task 5 feature metadata not found at {path}. Run the feature "
            "engineering notebook (docs/notebooks/task5_feature_engineering.ipynb) "
            "before training Task 6 models."
        )
    with open(path) as f:
        raw = json.load(f)
    return FeatureContract(raw=raw)


def load_feature_matrix(path: Path = TASK5_FEATURES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FeatureContractError(
            f"Task 5 feature matrix not found at {path}. Run the feature "
            "engineering notebook before training Task 6 models."
        )
    return pd.read_csv(path)
