"""Splitting must be chronological: no future lap ever leaks into a training
set, and the final holdout always sits strictly after development laps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.intelligence.ml.splits import chronological_holdout, expanding_window_folds


def _synthetic_panel(n_laps=40, n_drivers=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lap in range(1, n_laps + 1):
        for driver in range(n_drivers):
            rows.append({"LapNumber": lap, "Driver": driver, "value": rng.normal()})
    return pd.DataFrame(rows).reset_index(drop=True)


def test_holdout_test_laps_are_strictly_later_than_dev_laps():
    df = _synthetic_panel()
    h = chronological_holdout(df, test_fraction=0.2)
    assert max(h.dev_laps) < min(h.test_laps)


def test_holdout_keeps_whole_laps_together():
    df = _synthetic_panel()
    h = chronological_holdout(df, test_fraction=0.2)
    dev_laps_in_data = set(df.loc[h.dev_index, "LapNumber"])
    test_laps_in_data = set(df.loc[h.test_index, "LapNumber"])
    assert dev_laps_in_data == set(h.dev_laps)
    assert test_laps_in_data == set(h.test_laps)
    assert dev_laps_in_data.isdisjoint(test_laps_in_data)


def test_holdout_covers_every_row_exactly_once():
    df = _synthetic_panel()
    h = chronological_holdout(df, test_fraction=0.2)
    all_idx = np.sort(np.concatenate([h.dev_index, h.test_index]))
    assert list(all_idx) == list(df.index)


def test_expanding_window_folds_never_train_on_future_laps():
    df = _synthetic_panel()
    h = chronological_holdout(df, test_fraction=0.2)
    dev_df = df.loc[h.dev_index].reset_index(drop=True)
    folds = expanding_window_folds(dev_df, n_folds=3)

    for fold in folds:
        max_train_lap = dev_df.loc[fold.train_index, "LapNumber"].max()
        min_val_lap = dev_df.loc[fold.val_index, "LapNumber"].min()
        assert max_train_lap < min_val_lap, "a validation lap leaked into training"


def test_expanding_window_folds_are_deterministic():
    df = _synthetic_panel()
    h = chronological_holdout(df, test_fraction=0.2)
    dev_df = df.loc[h.dev_index].reset_index(drop=True)
    folds_a = expanding_window_folds(dev_df, n_folds=3)
    folds_b = expanding_window_folds(dev_df, n_folds=3)
    for fa, fb in zip(folds_a, folds_b):
        assert list(fa.train_index) == list(fb.train_index)
        assert list(fa.val_index) == list(fb.val_index)


def test_expanding_window_folds_train_set_grows_monotonically():
    df = _synthetic_panel()
    h = chronological_holdout(df, test_fraction=0.2)
    dev_df = df.loc[h.dev_index].reset_index(drop=True)
    folds = expanding_window_folds(dev_df, n_folds=3)
    sizes = [len(f.train_index) for f in folds]
    assert sizes == sorted(sizes)


def test_expanding_window_folds_raises_when_not_enough_laps():
    df = _synthetic_panel(n_laps=3, n_drivers=2)
    with pytest.raises(ValueError):
        expanding_window_folds(df, n_folds=5)
