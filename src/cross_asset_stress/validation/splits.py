"""Chronological, purged, and embargoed time-series splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    """Indices and metadata for a temporal validation window."""

    train_index: np.ndarray
    test_index: np.ndarray
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_horizon: int = 0
    embargo: int = 0


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_col: str = "forecast_timestamp",
    train_fraction: float = 0.70,
) -> SplitWindow:
    """Single chronological train/test split."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered = frame.sort_values(time_col).reset_index(drop=True)
    split_at = int(len(ordered) * train_fraction)
    if split_at <= 0 or split_at >= len(ordered):
        raise ValueError("Not enough rows for the requested chronological split")
    train_index = np.arange(0, split_at)
    test_index = np.arange(split_at, len(ordered))
    times = pd.to_datetime(ordered[time_col], utc=True)
    return SplitWindow(
        train_index=train_index,
        test_index=test_index,
        train_start=times.iloc[train_index[0]],
        train_end=times.iloc[train_index[-1]],
        test_start=times.iloc[test_index[0]],
        test_end=times.iloc[test_index[-1]],
    )


def purged_time_series_split(
    timestamps: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    n_splits: int = 3,
    horizon: int = 20,
    embargo: int = 0,
    min_train_size: int | None = None,
) -> Iterator[SplitWindow]:
    """Yield purged CV folds for forward-label time-series experiments.

    A training observation at row ``i`` is removed when its outcome window
    ``[i + 1, i + horizon]`` overlaps the validation window. Rows immediately
    after validation are also removed according to ``embargo``.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if horizon < 0 or embargo < 0:
        raise ValueError("horizon and embargo must be non-negative")

    ts = pd.Series(pd.to_datetime(timestamps, utc=True)).sort_values().reset_index(drop=True)
    n_obs = len(ts)
    if n_obs < n_splits + 2:
        raise ValueError("Not enough observations for requested splits")

    fold_sizes = np.full(n_splits, n_obs // n_splits, dtype=int)
    fold_sizes[: n_obs % n_splits] += 1
    starts = np.cumsum(np.r_[0, fold_sizes[:-1]])

    for fold, (start, size) in enumerate(zip(starts, fold_sizes, strict=True)):
        test_start_idx = int(start)
        test_end_idx = int(start + size - 1)
        test_index = np.arange(test_start_idx, test_end_idx + 1)

        all_indices = np.arange(n_obs)
        before_validation = all_indices + horizon < test_start_idx
        after_embargo = all_indices > test_end_idx + embargo
        train_index = all_indices[before_validation | after_embargo]

        if min_train_size is not None and len(train_index) < min_train_size:
            continue
        if len(train_index) == 0:
            continue

        train_times = ts.iloc[train_index]
        yield SplitWindow(
            train_index=train_index,
            test_index=test_index,
            train_start=train_times.iloc[0],
            train_end=train_times.iloc[-1],
            test_start=ts.iloc[test_start_idx],
            test_end=ts.iloc[test_end_idx],
            purge_horizon=horizon,
            embargo=embargo,
        )


def expanding_window_splits(
    timestamps: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
    horizon: int = 0,
    embargo: int = 0,
) -> Iterator[SplitWindow]:
    """Yield expanding-window splits with optional purging and embargo."""

    ts = pd.Series(pd.to_datetime(timestamps, utc=True)).sort_values().reset_index(drop=True)
    n_obs = len(ts)
    step = step_size or test_size
    start = min_train_size
    while start < n_obs:
        end = min(start + test_size, n_obs)
        test_index = np.arange(start, end)
        train_before = np.arange(0, max(0, start - horizon))
        train_after = np.arange(min(n_obs, end + embargo + 1), n_obs)
        train_index = np.r_[train_before, train_after]
        if len(train_index):
            yield SplitWindow(
                train_index=train_index,
                test_index=test_index,
                train_start=ts.iloc[train_index[0]],
                train_end=ts.iloc[train_index[-1]],
                test_start=ts.iloc[test_index[0]],
                test_end=ts.iloc[test_index[-1]],
                purge_horizon=horizon,
                embargo=embargo,
            )
        start += step

