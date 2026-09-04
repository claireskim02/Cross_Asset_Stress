"""Macro feature placeholders."""

from __future__ import annotations

import pandas as pd


def yield_curve_slope(long_rate: pd.Series, short_rate: pd.Series) -> pd.Series:
    """Long minus short rate."""

    return long_rate.astype(float) - short_rate.astype(float)


def release_surprise(actual: pd.Series, consensus: pd.Series) -> pd.Series:
    """Actual release value minus the point-in-time consensus."""

    return actual.astype(float) - consensus.astype(float)

