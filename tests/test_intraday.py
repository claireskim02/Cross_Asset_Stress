from __future__ import annotations

import numpy as np
import pandas as pd

from chronoswan.data.intraday import (
    build_es_impulse_events,
    build_intraday_return_panel,
    canonical_intraday_ticker,
    load_bloomberg_intraday_workbook,
)
from chronoswan.experiments.intraday_impulse_pca import (
    build_conditional_correlation_table,
    run_conditional_pca_suite,
)


def test_canonical_intraday_ticker_handles_bloomberg_suffixes() -> None:
    assert canonical_intraday_ticker("ES1 Index") == "ES1"
    assert canonical_intraday_ticker("TY1 Commodity") == "TY1"
    assert canonical_intraday_ticker("DXY  Currency") == "DXY"
    assert canonical_intraday_ticker("HYG US Equity") == "HYG"


def test_load_bloomberg_intraday_workbook_detects_offset_timestamp_column(tmp_path) -> None:
    path = tmp_path / "intraday.xlsx"
    normal = pd.DataFrame(
        {
            0: pd.date_range("2026-01-01 09:30", periods=4, freq="h"),
            1: [100, 101, 102, 103],
            2: [101, 102, 103, 104],
            3: [99, 100, 101, 102],
            4: [100.5, 101.5, 102.5, 103.5],
            5: [10, 11, 12, 13],
        }
    )
    offset = pd.DataFrame(
        {
            0: [np.nan, np.nan, "blank", "blank"],
            1: pd.date_range("2026-01-01 09:30", periods=4, freq="h"),
            2: [200, 201, 202, 203],
            3: [201, 202, 203, 204],
            4: [199, 200, 201, 202],
            5: [200.5, 201.5, 202.5, 203.5],
            6: [20, 21, 22, 23],
        }
    )
    with pd.ExcelWriter(path) as writer:
        normal.to_excel(writer, sheet_name="ES1 Index", header=False, index=False)
        offset.to_excel(writer, sheet_name="CL1 Commodity", header=False, index=False)

    loaded = load_bloomberg_intraday_workbook(path)

    assert set(loaded["ticker"]) == {"ES1", "CL1"}
    assert loaded.loc[loaded["ticker"] == "CL1", "close"].tolist() == [
        200.5,
        201.5,
        202.5,
        203.5,
    ]


def test_impulse_correlations_and_pca_on_synthetic_intraday_frame() -> None:
    timestamps = pd.date_range("2026-01-01", periods=180, freq="h")
    es = np.r_[np.repeat(0.001, 120), np.linspace(-0.02, -0.01, 10), np.repeat(0.001, 50)]
    nq = es * 1.2
    ty = -es * 0.7
    dxy = -es * 0.2
    long = pd.concat(
        [
            _make_ticker_frame(timestamps, "ES1", es),
            _make_ticker_frame(timestamps, "NQ1", nq),
            _make_ticker_frame(timestamps, "TY1", ty),
            _make_ticker_frame(timestamps, "DXY", dxy),
        ],
        ignore_index=True,
    )
    returns = build_intraday_return_panel(long)
    events = build_es_impulse_events(
        returns,
        rolling_window_days=2,
        impulse_quantile=0.90,
        min_window_bars=20,
    )

    correlations = build_conditional_correlation_table(returns, events)
    pca_summary, pca_loadings = run_conditional_pca_suite(
        returns,
        events,
        min_observations=5,
    )

    large_down = correlations.query("sample == 'large_down'")
    assert not large_down.empty
    assert large_down["n_obs"].max() >= 5
    assert not pca_summary.query("status == 'fit'").empty
    assert not pca_loadings.empty


def _make_ticker_frame(timestamps: pd.DatetimeIndex, ticker: str, returns: np.ndarray) -> pd.DataFrame:
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "ticker": ticker,
            "sheet_name": f"{ticker} Index",
            "asset_class": "test",
            "role": "test",
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1,
        }
    )
