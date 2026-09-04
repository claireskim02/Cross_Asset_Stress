"""Futures feature placeholders."""

from __future__ import annotations

import pandas as pd


def futures_basis(futures_price: pd.Series, spot_price: pd.Series) -> pd.Series:
    """Simple futures basis relative to spot."""

    return futures_price.astype(float) / spot_price.astype(float) - 1.0

