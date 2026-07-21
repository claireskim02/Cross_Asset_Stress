"""Liquidity feature placeholders."""

from __future__ import annotations

import pandas as pd


def rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    """Trailing z-score with no forward information."""

    rolling = values.astype(float).rolling(window, min_periods=max(5, window // 4))
    return (values - rolling.mean()) / rolling.std()

