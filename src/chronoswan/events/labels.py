"""Configurable event labels for extreme market-risk targets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_to_name(threshold: float) -> str:
    """Format a numeric threshold as a stable column fragment."""

    return f"{int(round(threshold * 100))}pct"


def forward_drawdown(prices: pd.Series, horizon: int) -> pd.Series:
    """Minimum forward close-to-close return over the next ``horizon`` rows."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future_prices = pd.concat([prices.shift(-step) for step in range(1, horizon + 1)], axis=1)
    future_min = future_prices.min(axis=1)
    return future_min / prices - 1.0


def forward_window_max(values: pd.Series, horizon: int) -> pd.Series:
    """Maximum value observed over the next ``horizon`` rows."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future_values = pd.concat([values.shift(-step) for step in range(1, horizon + 1)], axis=1)
    return future_values.max(axis=1)


def make_forward_drawdown_label(
    market_frame: pd.DataFrame,
    *,
    price_col: str = "spx_price",
    horizon: int = 20,
    threshold: float = 0.05,
) -> pd.Series:
    """Label rows where forward drawdown breaches a predeclared threshold."""

    drawdown = forward_drawdown(market_frame[price_col].astype(float), horizon=horizon)
    valid = market_frame[price_col].shift(-horizon).notna()
    label = (drawdown <= -abs(threshold)).astype(float)
    label.loc[~valid] = np.nan
    return label


def classify_event_phases(label: pd.Series) -> pd.Series:
    """Classify onset, continuation, recovery, calm, and unavailable rows."""

    phases: list[str] = []
    previous = 0
    for value in label:
        if pd.isna(value):
            phases.append("insufficient_forward_window")
            previous = 0
            continue
        current = int(value)
        if current == 1 and previous == 0:
            phases.append("onset")
        elif current == 1 and previous == 1:
            phases.append("continuation")
        elif current == 0 and previous == 1:
            phases.append("recovery")
        else:
            phases.append("calm")
        previous = current
    return pd.Series(phases, index=label.index, name="event_phase")


def event_run_ids(label: pd.Series) -> pd.Series:
    """Assign an event ID to each contiguous event run."""

    event_id = 0
    previous = 0
    ids: list[float] = []
    for value in label:
        current = 0 if pd.isna(value) else int(value)
        if current == 1 and previous == 0:
            event_id += 1
        ids.append(float(event_id) if current == 1 else np.nan)
        previous = current
    return pd.Series(ids, index=label.index, name="event_id")


def time_to_next_onset(label: pd.Series) -> pd.Series:
    """Business-row distance to the next event onset."""

    values = label.fillna(0).astype(int).to_numpy()
    previous = np.r_[0, values[:-1]]
    onset_indices = np.flatnonzero((values == 1) & (previous == 0))
    distances = np.full(len(values), np.nan)
    if onset_indices.size == 0:
        return pd.Series(distances, index=label.index, name="time_to_next_event")

    pointer = 0
    for idx in range(len(values)):
        while pointer < len(onset_indices) and onset_indices[pointer] < idx:
            pointer += 1
        if pointer < len(onset_indices):
            distances[idx] = onset_indices[pointer] - idx
    return pd.Series(distances, index=label.index, name="time_to_next_event")


def build_event_labels(
    market_frame: pd.DataFrame,
    *,
    horizon: int = 20,
    drawdown_threshold: float = 0.05,
    volatility_quantile: float = 0.95,
    vix_threshold: float = 35.0,
    price_col: str = "spx_price",
    realized_vol_col: str = "realized_vol_20",
    vix_col: str = "vix_like",
    credit_col: str = "credit_spread_like",
    liquidity_col: str = "liquidity_stress_like",
) -> pd.DataFrame:
    """Build the first set of predeclared synthetic event labels."""

    required = ["event_time", "forecast_timestamp", price_col, realized_vol_col, vix_col]
    missing = [column for column in required if column not in market_frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for labels: {missing}")

    frame = market_frame.sort_values("event_time").reset_index(drop=True).copy()
    drawdown = forward_drawdown(frame[price_col].astype(float), horizon=horizon)
    drawdown_col = f"drawdown_{threshold_to_name(drawdown_threshold)}_h{horizon}"
    drawdown_label = make_forward_drawdown_label(
        frame,
        price_col=price_col,
        horizon=horizon,
        threshold=drawdown_threshold,
    )

    vol_threshold = frame[realized_vol_col].quantile(volatility_quantile)
    future_vol_max = forward_window_max(frame[realized_vol_col].astype(float), horizon=horizon)
    vol_col = f"realized_vol_top_{int(round((1 - volatility_quantile) * 100))}pct_h{horizon}"
    vol_label = (future_vol_max >= vol_threshold).astype(float)
    vol_label.loc[frame[realized_vol_col].shift(-horizon).isna()] = np.nan

    future_vix_max = forward_window_max(frame[vix_col].astype(float), horizon=horizon)
    vix_label_col = f"vix_gt_{int(vix_threshold)}_h{horizon}"
    vix_label = (future_vix_max >= vix_threshold).astype(float)
    vix_label.loc[frame[vix_col].shift(-horizon).isna()] = np.nan

    credit_threshold = frame[credit_col].quantile(0.90) if credit_col in frame else np.inf
    liquidity_threshold = frame[liquidity_col].quantile(0.90) if liquidity_col in frame else np.inf
    future_credit = (
        forward_window_max(frame[credit_col].astype(float), horizon=horizon)
        if credit_col in frame
        else pd.Series(np.nan, index=frame.index)
    )
    future_liquidity = (
        forward_window_max(frame[liquidity_col].astype(float), horizon=horizon)
        if liquidity_col in frame
        else pd.Series(np.nan, index=frame.index)
    )
    joint_col = f"joint_stress_h{horizon}"
    joint_label = (
        (drawdown_label == 1)
        & ((vol_label == 1) | (vix_label == 1))
        & ((future_credit >= credit_threshold) | (future_liquidity >= liquidity_threshold))
    ).astype(float)
    joint_label.loc[drawdown_label.isna()] = np.nan

    phases = classify_event_phases(drawdown_label)
    ids = event_run_ids(drawdown_label)
    overlap = (drawdown_label.fillna(0).astype(int) == 1) & phases.eq("continuation")

    return pd.DataFrame(
        {
            "event_time": frame["event_time"],
            "forecast_timestamp": frame["forecast_timestamp"],
            "forecast_horizon_days": horizon,
            "forward_drawdown": drawdown,
            drawdown_col: drawdown_label,
            vol_col: vol_label,
            vix_label_col: vix_label,
            joint_col: joint_label,
            "event_phase": phases,
            "event_id": ids,
            "overlapping_event_window": overlap,
            "time_to_next_event": time_to_next_onset(drawdown_label),
        }
    )

