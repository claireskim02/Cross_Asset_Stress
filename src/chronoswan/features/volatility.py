"""Volatility feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series, window: int, annualization: int = 252) -> pd.Series:
    """Trailing realized volatility."""

    return returns.rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(annualization)


def downside_semivariance(returns: pd.Series, window: int) -> pd.Series:
    """Trailing downside semivariance."""

    downside = returns.clip(upper=0)
    return downside.pow(2).rolling(window, min_periods=max(5, window // 4)).mean()

