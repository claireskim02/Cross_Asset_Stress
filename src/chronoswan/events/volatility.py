"""Volatility-regime helpers."""

from __future__ import annotations

import pandas as pd

from chronoswan.events.labels import forward_window_max


def forward_top_quantile_label(values: pd.Series, horizon: int, quantile: float) -> pd.Series:
    """Label rows where a future window reaches a predeclared quantile threshold."""

    threshold = values.quantile(quantile)
    return (forward_window_max(values, horizon=horizon) >= threshold).astype(float)

