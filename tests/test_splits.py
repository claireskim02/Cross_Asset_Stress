from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_stress.validation.splits import chronological_split, purged_time_series_split


def test_chronological_split_orders_indices() -> None:
    frame = pd.DataFrame({"forecast_timestamp": pd.bdate_range("2020-01-01", periods=10, tz="UTC")})

    split = chronological_split(frame, train_fraction=0.6)

    assert split.train_index.tolist() == list(range(6))
    assert split.test_index.tolist() == list(range(6, 10))


def test_purged_split_removes_overlapping_forward_windows_and_embargo() -> None:
    timestamps = pd.bdate_range("2020-01-01", periods=90, tz="UTC")
    horizon = 5
    embargo = 2

    splits = list(purged_time_series_split(timestamps, n_splits=3, horizon=horizon, embargo=embargo))

    assert splits
    for split in splits:
        test_start = int(split.test_index[0])
        test_end = int(split.test_index[-1])
        assert not np.intersect1d(split.train_index, split.test_index).size
        assert all((idx + horizon < test_start) or (idx > test_end + embargo) for idx in split.train_index)

