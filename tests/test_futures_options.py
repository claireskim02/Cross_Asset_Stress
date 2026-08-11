import pandas as pd

from chronoswan.data.futures_options import (
    build_constant_tenor_ivm_features,
    summarize_ivm_pull,
)


def test_build_constant_tenor_ivm_features_selects_nearest_expiry() -> None:
    ivm = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "research_ticker": "ES",
                "days_expiration": 20.0,
                "atm": 0.10,
                "rr25": -0.01,
                "rr10": -0.02,
                "fly25": 0.01,
                "fly10": 0.02,
            },
            {
                "date": "2024-01-02",
                "research_ticker": "ES",
                "days_expiration": 32.0,
                "atm": 0.20,
                "rr25": -0.03,
                "rr10": -0.04,
                "fly25": 0.03,
                "fly10": 0.04,
            },
            {
                "date": "2024-01-02",
                "research_ticker": "TY",
                "days_expiration": 31.0,
                "atm": 0.30,
                "rr25": 0.01,
                "rr10": 0.02,
                "fly25": 0.01,
                "fly10": 0.02,
            },
        ]
    )

    features = build_constant_tenor_ivm_features(ivm, target_days=(30,), max_distance_days=15)

    assert features.loc[pd.Timestamp("2024-01-02"), "ES_atm_30d"] == 0.20
    assert features.loc[pd.Timestamp("2024-01-02"), "ES_rr25_30d"] == -0.03
    assert features.loc[pd.Timestamp("2024-01-02"), "TY_atm_30d"] == 0.30


def test_summarize_ivm_pull_handles_empty_frame() -> None:
    summary = summarize_ivm_pull(pd.DataFrame())

    assert list(summary.columns) == [
        "research_ticker",
        "rows",
        "start",
        "end",
        "unique_dates",
        "min_dte",
        "median_dte",
        "max_dte",
    ]
    assert summary.empty

