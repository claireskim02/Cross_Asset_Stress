"""Return feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def close_to_close_returns(price: pd.Series) -> pd.Series:
    """Log close-to-close returns."""

    return np.log(price.astype(float)).diff()


def rolling_momentum(returns: pd.Series, window: int) -> pd.Series:
    """Trailing cumulative log return."""

    return returns.rolling(window, min_periods=max(2, window // 4)).sum()

