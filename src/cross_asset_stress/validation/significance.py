"""Statistical comparison placeholders."""

from __future__ import annotations

import numpy as np


def paired_brier_difference(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Mean Brier loss of model A minus model B."""

    return float(np.mean((y_true - p_a) ** 2 - (y_true - p_b) ** 2))

