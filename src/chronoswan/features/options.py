"""Options feature placeholders."""

from __future__ import annotations

import pandas as pd


def implied_realized_spread(implied_vol: pd.Series, realized_vol: pd.Series) -> pd.Series:
    """Difference between implied and realized volatility."""

    return implied_vol.astype(float) - realized_vol.astype(float)


def skew_change(skew: pd.Series, periods: int = 1) -> pd.Series:
    """Change in an implied-skew measure."""

    return skew.astype(float).diff(periods)

