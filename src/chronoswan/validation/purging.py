"""Purging helpers for forward-label validation."""

from __future__ import annotations

import numpy as np


def purged_train_mask(
    n_obs: int,
    *,
    test_start_idx: int,
    test_end_idx: int,
    horizon: int,
    embargo: int = 0,
) -> np.ndarray:
    """Boolean mask of rows allowed in training around a validation interval."""

    indices = np.arange(n_obs)
    before_validation = indices + horizon < test_start_idx
    after_embargo = indices > test_end_idx + embargo
    return before_validation | after_embargo

