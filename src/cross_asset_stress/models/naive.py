"""Naive probability benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd


class HistoricalFrequencyModel:
    """Unconditional historical event-frequency benchmark."""

    def __init__(self) -> None:
        self.probability_: float | None = None

    def fit(self, y: pd.Series | np.ndarray) -> "HistoricalFrequencyModel":
        target = pd.Series(y).dropna().astype(float)
        if target.empty:
            raise ValueError("Cannot fit HistoricalFrequencyModel on an empty target")
        self.probability_ = float(target.mean())
        return self

    def predict_proba(self, n_obs: int) -> np.ndarray:
        if self.probability_ is None:
            raise RuntimeError("Model is not fitted")
        return np.full(n_obs, self.probability_, dtype=float)


class RollingHistoricalFrequencyModel:
    """Rolling event-frequency benchmark using only prior observations."""

    def __init__(self, window: int = 252, min_periods: int = 20) -> None:
        self.window = window
        self.min_periods = min_periods

    def predict_in_sample(self, y: pd.Series) -> pd.Series:
        target = pd.Series(y).astype(float)
        return target.shift(1).rolling(self.window, min_periods=self.min_periods).mean()

