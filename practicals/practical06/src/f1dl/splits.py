"""
f1dl.splits
===========

Time-aware validation for the Task 5 driver x lap panel.

This module is copied verbatim from Task 6's ``app.intelligence.ml.splits``
so Task 7 is validated on *identical* folds and an identical untouched test
set - any DL-vs-classical difference is then attributable to the model, not
to a different validation scheme. (On the ``proj-mode`` branch the unified
app imports this module directly instead of copying it.)

This is a driver x lap panel ordered by ``LapNumber``: every driver is
observed once per lap (see
``feature_metadata.json``: ``preprocessing_contract.validation`` =
"Expanding-window lap-forward split; keep whole laps in one fold. Do not
use random K-fold - this is a time-ordered panel.").

Two mechanisms are implemented here, both keyed on whole lap numbers so no
fold ever splits a lap across train/validation:

``chronological_holdout``
    Reserves the chronologically last block of laps as a final test set that
    stays untouched until model selection is complete.

``expanding_window_folds``
    Builds deterministic lap-forward CV folds over the remaining
    (development) laps: fold *i* trains on every lap up to and including
    block *i-1* and validates on block *i*. Block boundaries are computed
    generically from the number of requested folds — they are not hand
    positioned around any particular event, so they cannot be accused of
    being tuned to flatter a metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HoldoutSplit:
    dev_index: np.ndarray
    test_index: np.ndarray
    dev_laps: list[int]
    test_laps: list[int]

    def to_metadata(self) -> dict:
        return {
            "dev_laps": self.dev_laps,
            "test_laps": self.test_laps,
            "dev_rows": int(len(self.dev_index)),
            "test_rows": int(len(self.test_index)),
        }


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_index: np.ndarray
    val_index: np.ndarray
    train_laps: list[int]
    val_laps: list[int]

    def to_metadata(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_laps": self.train_laps,
            "val_laps": self.val_laps,
            "train_rows": int(len(self.train_index)),
            "val_rows": int(len(self.val_index)),
        }


def chronological_holdout(
    df: pd.DataFrame, lap_col: str = "LapNumber", test_fraction: float = 0.2
) -> HoldoutSplit:
    """Reserve the chronologically last ``test_fraction`` of laps as a final,
    untouched test set. Whole laps are kept together on one side of the
    boundary."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    unique_laps = np.sort(df[lap_col].unique())
    n_test_laps = max(1, round(len(unique_laps) * test_fraction))
    dev_laps = unique_laps[:-n_test_laps]
    test_laps = unique_laps[-n_test_laps:]

    dev_index = df.index[df[lap_col].isin(dev_laps)].to_numpy()
    test_index = df.index[df[lap_col].isin(test_laps)].to_numpy()

    return HoldoutSplit(
        dev_index=dev_index,
        test_index=test_index,
        dev_laps=[int(x) for x in dev_laps],
        test_laps=[int(x) for x in test_laps],
    )


def expanding_window_folds(
    df: pd.DataFrame, lap_col: str = "LapNumber", n_folds: int = 4
) -> list[Fold]:
    """Deterministic lap-forward expanding-window CV folds.

    Unique laps (already restricted to the development set by the caller)
    are cut into ``n_folds + 1`` contiguous, near-equal blocks. Fold *i*
    (1-indexed) trains on blocks ``0..i-1`` and validates on block ``i``.
    Block 0 is never used for validation on its own — it only ever
    contributes training rows — so every fold has a non-empty training set.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    unique_laps = np.sort(df[lap_col].unique())
    n_blocks = n_folds + 1
    if len(unique_laps) < n_blocks:
        raise ValueError(
            f"Not enough distinct laps ({len(unique_laps)}) to build "
            f"{n_folds} expanding-window folds (needs >= {n_blocks})."
        )

    blocks = [b for b in np.array_split(unique_laps, n_blocks) if len(b) > 0]

    folds: list[Fold] = []
    for i in range(1, len(blocks)):
        train_laps = np.concatenate(blocks[:i])
        val_laps = blocks[i]

        train_index = df.index[df[lap_col].isin(train_laps)].to_numpy()
        val_index = df.index[df[lap_col].isin(val_laps)].to_numpy()

        folds.append(
            Fold(
                fold_id=i,
                train_index=train_index,
                val_index=val_index,
                train_laps=[int(x) for x in train_laps],
                val_laps=[int(x) for x in val_laps],
            )
        )
    return folds
