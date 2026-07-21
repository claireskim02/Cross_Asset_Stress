"""Rare-event probability metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def evaluate_probabilities(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray) -> dict[str, float]:
    """Evaluate binary probabilistic forecasts with rare-event-safe metrics."""

    y = pd.Series(y_true).reset_index(drop=True).astype(float)
    p = pd.Series(y_prob).reset_index(drop=True).astype(float).clip(1e-6, 1 - 1e-6)
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int)
    p = p.loc[valid]
    if len(y) == 0:
        raise ValueError("No valid rows for evaluation")

    metrics = {
        "n_obs": float(len(y)),
        "positive_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    if y.nunique() == 2:
        metrics["average_precision"] = float(average_precision_score(y, p))
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    else:
        metrics["average_precision"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Simple equal-width expected calibration error."""

    y = pd.Series(y_true).reset_index(drop=True).astype(float)
    p = pd.Series(y_prob).reset_index(drop=True).astype(float).clip(0, 1)
    valid = y.notna() & p.notna()
    y = y.loc[valid]
    p = p.loc[valid]
    bins = pd.cut(p, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    ece = 0.0
    for _, idx in p.groupby(bins, observed=False).groups.items():
        if len(idx) == 0:
            continue
        bin_y = y.loc[idx]
        bin_p = p.loc[idx]
        ece += len(idx) / len(p) * abs(float(bin_y.mean()) - float(bin_p.mean()))
    return float(ece)
