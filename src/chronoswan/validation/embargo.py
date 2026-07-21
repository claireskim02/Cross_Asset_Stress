"""Embargo helpers."""

from __future__ import annotations

import numpy as np


def embargo_mask(n_obs: int, *, test_end_idx: int, embargo: int) -> np.ndarray:
    """Return true for rows outside the post-validation embargo interval."""

    indices = np.arange(n_obs)
    return ~((indices > test_end_idx) & (indices <= test_end_idx + embargo))

