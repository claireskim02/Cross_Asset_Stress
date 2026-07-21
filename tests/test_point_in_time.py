from __future__ import annotations

import pandas as pd
import pytest

from chronoswan.data.point_in_time import (
    FutureDataError,
    as_of_join,
    build_point_in_time_feature_matrix,
    validate_no_future_availability,
)


def test_as_of_join_uses_only_available_records() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_timestamp": pd.to_datetime(
                ["2020-01-03 00:00:00Z", "2020-01-06 00:00:00Z"]
            )
        }
    )
    records = pd.DataFrame(
        {
            "earliest_valid_prediction_timestamp": pd.to_datetime(
                ["2020-01-02 00:00:00Z", "2020-01-05 00:00:00Z"]
            ),
            "value": [1.0, 2.0],
        }
    )

    joined = as_of_join(forecasts, records)

    assert joined["value"].tolist() == [1.0, 2.0]
    assert (
        joined["earliest_valid_prediction_timestamp"] <= joined["forecast_timestamp"]
    ).all()


def test_feature_matrix_does_not_pull_future_feature() -> None:
    feature_store = pd.DataFrame(
        {
            "feature_name": ["vix_like", "vix_like"],
            "value": [20.0, 99.0],
            "earliest_valid_prediction_timestamp": pd.to_datetime(
                ["2020-01-02 00:00:00Z", "2020-01-10 00:00:00Z"]
            ),
        }
    )

    matrix = build_point_in_time_feature_matrix(
        [pd.Timestamp("2020-01-05", tz="UTC")],
        feature_store,
        feature_names=["vix_like"],
    )

    assert matrix.loc[0, "vix_like"] == 20.0


def test_validate_no_future_availability_rejects_joined_future_rows() -> None:
    joined = pd.DataFrame(
        {
            "forecast_timestamp": pd.to_datetime(["2020-01-03 00:00:00Z"]),
            "vix_like__available_at": pd.to_datetime(["2020-01-04 00:00:00Z"]),
        }
    )

    with pytest.raises(FutureDataError):
        validate_no_future_availability(joined, availability_cols=["vix_like__available_at"])

