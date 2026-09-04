"""Point-in-time joins and leakage guards."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


class PointInTimeError(ValueError):
    """Base class for point-in-time contract violations."""


class FutureDataError(PointInTimeError):
    """Raised when data availability is later than the forecast timestamp."""


def _coerce_timestamp_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing timestamp column: {column}")
    return pd.to_datetime(frame[column], utc=True)


def validate_no_future_availability(
    frame: pd.DataFrame,
    forecast_time_col: str = "forecast_timestamp",
    availability_cols: Iterable[str] = ("earliest_valid_prediction_timestamp",),
) -> None:
    """Raise if any availability timestamp is after its forecast timestamp."""

    if frame.empty:
        return
    forecast_ts = _coerce_timestamp_column(frame, forecast_time_col)
    violations: list[str] = []
    for column in availability_cols:
        if column not in frame.columns:
            continue
        available_ts = pd.to_datetime(frame[column], utc=True)
        mask = available_ts.notna() & (available_ts > forecast_ts)
        if mask.any():
            violations.append(f"{column}: {int(mask.sum())} rows")
    if violations:
        raise FutureDataError(
            "Future data was joined into a forecast frame: " + ", ".join(violations)
        )


def as_of_join(
    forecasts: pd.DataFrame,
    records: pd.DataFrame,
    *,
    forecast_time_col: str = "forecast_timestamp",
    availability_time_col: str = "earliest_valid_prediction_timestamp",
    by: Sequence[str] | None = None,
    direction: str = "backward",
    suffix: str = "_pit",
    strict: bool = True,
) -> pd.DataFrame:
    """Join each forecast row to the latest record available at that time.

    The function uses the record availability timestamp, not the event timestamp,
    as the merge clock. With ``direction='backward'``, records whose availability
    is later than the forecast timestamp are never returned.
    """

    if direction != "backward":
        raise ValueError("Cross-Asset Stress Monitor as_of_join only supports backward point-in-time joins")
    if forecasts.empty:
        return forecasts.copy()
    if records.empty:
        return forecasts.copy()

    by_cols = list(by or [])
    left = forecasts.copy()
    right = records.copy()
    left[forecast_time_col] = _coerce_timestamp_column(left, forecast_time_col)
    right[availability_time_col] = _coerce_timestamp_column(right, availability_time_col)

    left = left.sort_values(by_cols + [forecast_time_col])
    right = right.sort_values(by_cols + [availability_time_col])
    merged = pd.merge_asof(
        left,
        right,
        left_on=forecast_time_col,
        right_on=availability_time_col,
        by=by_cols or None,
        direction="backward",
        suffixes=("", suffix),
    )

    if strict:
        validate_no_future_availability(
            merged,
            forecast_time_col=forecast_time_col,
            availability_cols=(availability_time_col,),
        )
    return merged


def build_point_in_time_feature_matrix(
    forecast_timestamps: Sequence[pd.Timestamp] | pd.Series,
    feature_store: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
    forecast_time_col: str = "forecast_timestamp",
    availability_time_col: str = "earliest_valid_prediction_timestamp",
    feature_name_col: str = "feature_name",
    value_col: str = "value",
    strict: bool = True,
) -> pd.DataFrame:
    """Build a wide feature matrix using point-in-time availability."""

    features = list(feature_names or sorted(feature_store[feature_name_col].dropna().unique()))
    forecasts = pd.DataFrame({forecast_time_col: pd.to_datetime(forecast_timestamps, utc=True)})
    forecasts = forecasts.drop_duplicates().sort_values(forecast_time_col).reset_index(drop=True)
    output = forecasts.copy()

    for feature in features:
        subset = feature_store.loc[feature_store[feature_name_col] == feature].copy()
        if subset.empty:
            output[feature] = pd.NA
            output[f"{feature}__available_at"] = pd.NaT
            continue
        subset = subset[[availability_time_col, value_col]].sort_values(availability_time_col)
        joined = pd.merge_asof(
            forecasts,
            subset,
            left_on=forecast_time_col,
            right_on=availability_time_col,
            direction="backward",
        )
        output[feature] = joined[value_col]
        output[f"{feature}__available_at"] = joined[availability_time_col]

    if strict:
        availability_cols = [f"{feature}__available_at" for feature in features]
        validate_no_future_availability(
            output,
            forecast_time_col=forecast_time_col,
            availability_cols=availability_cols,
        )
    return output

