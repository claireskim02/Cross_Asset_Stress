"""Calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def calibration_slope_intercept(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray) -> dict[str, float]:
    """Estimate calibration slope and intercept via logistic recalibration."""

    y = pd.Series(y_true).reset_index(drop=True).astype(float)
    p = pd.Series(y_prob).reset_index(drop=True).astype(float).clip(1e-6, 1 - 1e-6)
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int)
    p = p.loc[valid]
    if y.nunique() < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    logits = np.log(p / (1 - p)).to_numpy().reshape(-1, 1)
    model = LogisticRegression(penalty=None, max_iter=1000).fit(logits, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }
