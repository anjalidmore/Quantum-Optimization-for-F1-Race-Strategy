"""
app.intelligence.ml.splits
============================

Time-aware validation for the Task 5 driver x lap panel.

This is a driver x lap panel ordered by ``LapNumber``: every driver is
observed once per lap, laps 4..55, ten drivers per lap (see
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


@dataclass(frozen=True)
class RaceHoldoutSplit:
    """A holdout that reserves whole *races*, not the tail of one race."""

    dev_index: np.ndarray
    test_index: np.ndarray
    dev_races: list
    test_races: list
    race_column: str

    def to_metadata(self) -> dict:
        return {
            "strategy": "race-level holdout",
            "race_column": self.race_column,
            "dev_races": [str(r) for r in self.dev_races],
            "test_races": [str(r) for r in self.test_races],
            "dev_rows": int(len(self.dev_index)),
            "test_rows": int(len(self.test_index)),
        }


def available_race_column(df: pd.DataFrame) -> str | None:
    """The column identifying which race a row came from, if the data has one.

    The single-session matrix this project currently trains on has no such
    column — every row is the same Grand Prix. Returns ``None`` in that case so
    callers can say so explicitly rather than inventing a grouping.
    """
    for candidate in ("raceId", "race_id", "RaceId", "Event", "event", "session_id"):
        if candidate in df.columns and df[candidate].nunique() > 1:
            return candidate
    return None


def race_level_holdout(
    df: pd.DataFrame,
    race_col: str | None = None,
    test_fraction: float = 0.2,
    lap_col: str = "LapNumber",
) -> RaceHoldoutSplit:
    """Hold out whole races, keeping every lap of a race on one side.

    **Why this exists.** The current ``chronological_holdout`` reserves the last
    20% of *laps within one race*. That answers "can the model extrapolate to
    the end of a race it has already seen most of?" — which is why Task 6's
    regression scored a negative test R² there: the closing laps are a different
    fuel and tyre regime from the laps it trained on.

    The question that actually matters for deployment is different: "can the
    model predict a race it has never seen?" Only a race-level holdout answers
    that, because a lap-level split leaks race-specific conditions (track,
    weather, tyre allocation) across the boundary.

    **Status: infrastructure, not yet exercised.** The committed dataset is a
    single session, so there is nothing to hold out at race level — this raises
    ``ValueError`` rather than silently degrading to something else. Fetching
    several sessions with ``scripts/fetch_real_session.py`` is what makes it
    usable, and is tracked separately in ``TODO.md``.

    Races are ordered by their earliest lap so the split stays chronological
    between races as well as within them.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    race_col = race_col or available_race_column(df)
    if race_col is None:
        raise ValueError(
            "No race identifier column with more than one distinct value was found, so a "
            "race-level holdout cannot be built — this dataset is a single session. Fetch "
            "additional sessions with scripts/fetch_real_session.py first. Use "
            "chronological_holdout() for single-session data, and read its test R² knowing "
            "it measures within-race extrapolation rather than transfer to a new race."
        )

    # Order races chronologically by their first lap, so the held-out races are
    # the later ones rather than an arbitrary subset.
    if lap_col in df.columns:
        order = df.groupby(race_col)[lap_col].min().sort_values().index.tolist()
    else:
        order = sorted(df[race_col].unique())

    n_test = max(1, round(len(order) * test_fraction))
    if n_test >= len(order):
        raise ValueError(
            f"test_fraction={test_fraction} would hold out every one of the {len(order)} "
            f"races, leaving nothing to train on."
        )

    dev_races, test_races = order[:-n_test], order[-n_test:]
    dev_index = df.index[df[race_col].isin(dev_races)].to_numpy()
    test_index = df.index[df[race_col].isin(test_races)].to_numpy()

    return RaceHoldoutSplit(
        dev_index=dev_index,
        test_index=test_index,
        dev_races=dev_races,
        test_races=test_races,
        race_column=race_col,
    )
