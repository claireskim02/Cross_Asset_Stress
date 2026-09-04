"""Synthetic point-in-time data for the first Cross-Asset Stress Monitor demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cross_asset_stress.data.schemas import FeatureRecord, SourceType
from cross_asset_stress.events.labels import build_event_labels


CLEAN_FEATURES = [
    "spx_return",
    "realized_vol_20",
    "downside_semivariance_20",
    "max_drawdown_60",
    "vix_like",
    "skew_like",
    "put_call_like",
    "credit_spread_like",
    "futures_basis_like",
    "liquidity_stress_like",
]

LEAKED_FEATURES = [
    "leaked_forward_drawdown_20",
    "leaked_future_stress_flag",
]


def _next_business_timestamp(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (index + pd.offsets.BDay(1)).normalize() + pd.Timedelta(hours=14, minutes=30)


def _future_window_stat(values: pd.Series, horizon: int, stat: str) -> pd.Series:
    future = pd.concat([values.shift(-step) for step in range(1, horizon + 1)], axis=1)
    if stat == "min":
        return future.min(axis=1)
    if stat == "max":
        return future.max(axis=1)
    raise ValueError(f"Unsupported future stat: {stat}")


def _build_latent_stress(n_obs: int, rng: np.random.Generator) -> np.ndarray:
    stress = np.zeros(n_obs, dtype=int)
    if n_obs < 260:
        return stress

    forced_starts = [int(n_obs * 0.18), int(n_obs * 0.43), int(n_obs * 0.68), int(n_obs * 0.84)]
    for start in forced_starts:
        duration = int(rng.integers(12, 42))
        stress[start : min(n_obs, start + duration)] = 1

    i = 60
    while i < n_obs - 80:
        if rng.random() < 0.0025:
            duration = int(rng.integers(8, 35))
            stress[i : min(n_obs, i + duration)] = 1
            i += duration + int(rng.integers(60, 180))
        i += 1
    return stress


def generate_synthetic_market_frame(
    start: str = "1995-01-02",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic wide market panel with deliberate leakage traps."""

    dates = pd.bdate_range(start=start, end=end, tz="UTC")
    if len(dates) == 0:
        raise ValueError("Synthetic date range produced no business days")

    rng = np.random.default_rng(seed)
    stress = _build_latent_stress(len(dates), rng)
    onset = (stress == 1) & (np.r_[0, stress[:-1]] == 0)

    base_vol = 0.008 + 0.024 * stress
    clustered_noise = rng.normal(0.0, 0.002, len(dates)).cumsum()
    clustered_noise = pd.Series(clustered_noise).rolling(20, min_periods=1).mean().to_numpy()
    daily_vol = np.clip(base_vol + np.abs(clustered_noise) * 0.05, 0.004, 0.075)

    returns = rng.normal(0.00025 - 0.0018 * stress, daily_vol)
    returns[onset] -= rng.uniform(0.02, 0.055, onset.sum())
    jump_days = (stress == 1) & (rng.random(len(dates)) < 0.09)
    returns[jump_days] -= rng.uniform(0.01, 0.045, jump_days.sum())

    returns_s = pd.Series(returns, index=dates)
    price = 100.0 * np.exp(returns_s.cumsum())
    realized_vol_20 = returns_s.rolling(20, min_periods=5).std().fillna(returns_s.expanding().std())
    realized_vol_20 = (realized_vol_20.fillna(returns_s.std()) * np.sqrt(252)).clip(lower=0.01)
    downside = returns_s.clip(upper=0)
    downside_semivariance_20 = (
        downside.pow(2).rolling(20, min_periods=5).mean().fillna(downside.pow(2).expanding().mean())
    )
    rolling_peak = price.rolling(60, min_periods=5).max()
    max_drawdown_60 = (price / rolling_peak - 1.0).fillna(0.0)
    recent_downside = (-returns_s.rolling(5, min_periods=1).sum()).clip(lower=0.0)

    vix_like = (100 * realized_vol_20 + 9.0 * stress + rng.normal(0, 1.8, len(dates))).clip(8, 85)
    skew_like = (112 + 13.0 * stress + 90 * recent_downside + rng.normal(0, 4.0, len(dates))).clip(
        95, 180
    )
    put_call_like = (0.72 + 0.45 * stress + 7.0 * recent_downside + rng.normal(0, 0.08, len(dates))).clip(
        0.35, 2.2
    )
    credit_spread_like = (
        85 + 120 * stress + 300 * recent_downside + rng.normal(0, 10, len(dates))
    ).clip(35, 650)
    futures_basis_like = (rng.normal(0.03, 0.10, len(dates)) - 0.22 * stress).clip(-1.2, 1.2)
    liquidity_stress_like = (0.18 + 0.58 * stress + 8.0 * recent_downside + rng.normal(0, 0.07, len(dates))).clip(
        0, 1
    )

    forward_min_price = _future_window_stat(price, horizon=20, stat="min")
    leaked_forward_drawdown_20 = (forward_min_price / price - 1.0).abs().fillna(0.0)
    future_stress = pd.concat(
        [pd.Series(stress, index=dates).shift(-step) for step in range(1, 21)], axis=1
    ).max(axis=1)

    event_time = dates.normalize()
    observation_time = event_time + pd.Timedelta(hours=21)
    release_time = event_time + pd.Timedelta(hours=22)
    ingestion_time = release_time + pd.Timedelta(minutes=30)
    forecast_timestamp = _next_business_timestamp(event_time)

    frame = pd.DataFrame(
        {
            "event_time": event_time,
            "forecast_timestamp": forecast_timestamp,
            "observation_time": observation_time,
            "release_time": release_time,
            "ingestion_time": ingestion_time,
            "earliest_valid_prediction_timestamp": forecast_timestamp,
            "source": "synthetic_market_v0",
            "vintage": "synthetic_v0",
            "spx_price": price.to_numpy(),
            "spx_return": returns_s.to_numpy(),
            "realized_vol_20": realized_vol_20.to_numpy(),
            "downside_semivariance_20": downside_semivariance_20.to_numpy(),
            "max_drawdown_60": max_drawdown_60.to_numpy(),
            "vix_like": np.asarray(vix_like),
            "skew_like": np.asarray(skew_like),
            "put_call_like": np.asarray(put_call_like),
            "credit_spread_like": np.asarray(credit_spread_like),
            "futures_basis_like": np.asarray(futures_basis_like),
            "liquidity_stress_like": np.asarray(liquidity_stress_like),
            "latent_stress_state": stress,
            "leaked_forward_drawdown_20": leaked_forward_drawdown_20.to_numpy(),
            "leaked_future_stress_flag": future_stress.fillna(0).astype(int).to_numpy(),
        }
    )
    return frame


def to_feature_store(market_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide synthetic market frame to long point-in-time feature records."""

    records: list[dict[str, object]] = []
    for row in market_frame.itertuples(index=False):
        row_data = row._asdict()
        for feature_name in CLEAN_FEATURES + LEAKED_FEATURES:
            is_leaked = feature_name in LEAKED_FEATURES
            record = FeatureRecord(
                event_time=row_data["event_time"],
                observation_time=row_data["observation_time"],
                release_time=row_data["release_time"],
                ingestion_time=row_data["ingestion_time"],
                earliest_valid_prediction_timestamp=row_data[
                    "earliest_valid_prediction_timestamp"
                ],
                source=row_data["source"],
                source_type=SourceType.SYNTHETIC,
                vintage=row_data["vintage"],
                transformation_window=_transformation_window(feature_name),
                symbol="SPX",
                feature_name=feature_name,
                value=float(row_data[feature_name]),
                is_known_leak=is_leaked,
                leak_reason=(
                    "Computed from future outcomes and included only for audit testing."
                    if is_leaked
                    else None
                ),
            )
            records.append(record.model_dump())
    return pd.DataFrame.from_records(records)


def _transformation_window(feature_name: str) -> str | None:
    if feature_name.endswith("_20"):
        return "20 business days"
    if feature_name.endswith("_60"):
        return "60 business days"
    if feature_name in {"leaked_future_stress_flag", "leaked_forward_drawdown_20"}:
        return "future 20 business days - invalid feature"
    return "current observation"


def generate_synthetic_feature_store(
    start: str = "1995-01-02",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Return long-form synthetic features using the point-in-time contract."""

    return to_feature_store(generate_synthetic_market_frame(start=start, end=end, seed=seed))


def write_synthetic_dataset(
    output_dir: str | Path = "data/synthetic",
    *,
    start: str = "1995-01-02",
    end: str = "2025-12-31",
    seed: int = 42,
) -> dict[str, Path]:
    """Generate and write synthetic features, market panel, and labels."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    market = generate_synthetic_market_frame(start=start, end=end, seed=seed)
    features = to_feature_store(market)
    labels = build_event_labels(market)

    paths = {
        "market": output_path / "synthetic_market.parquet",
        "features": output_path / "synthetic_pit_features.parquet",
        "labels": output_path / "synthetic_labels.parquet",
    }
    market.to_parquet(paths["market"], index=False)
    features.to_parquet(paths["features"], index=False)
    labels.to_parquet(paths["labels"], index=False)
    return paths

