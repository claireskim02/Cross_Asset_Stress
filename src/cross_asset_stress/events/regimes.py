"""Regime-state utilities."""

from __future__ import annotations

import pandas as pd


def regime_transition_table(phases: pd.Series) -> pd.DataFrame:
    """Return counts of event-state transitions."""

    current = phases.astype(str)
    previous = current.shift(1).fillna("start")
    return pd.crosstab(previous, current)

