"""Drawdown helpers."""

from __future__ import annotations

import pandas as pd

from cross_asset_stress.events.labels import forward_drawdown, make_forward_drawdown_label

__all__ = ["forward_drawdown", "make_forward_drawdown_label"]


def realized_drawdown(price: pd.Series, window: int) -> pd.Series:
    """Trailing drawdown from the rolling peak."""

    rolling_peak = price.rolling(window, min_periods=1).max()
    return price / rolling_peak - 1.0

