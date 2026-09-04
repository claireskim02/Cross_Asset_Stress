from __future__ import annotations

import pandas as pd

from cross_asset_stress.events.labels import build_event_labels, classify_event_phases, make_forward_drawdown_label


def test_forward_drawdown_label_detects_breach() -> None:
    frame = pd.DataFrame({"spx_price": [100, 98, 88, 90, 91]})

    label = make_forward_drawdown_label(frame, horizon=2, threshold=0.10)

    assert label.iloc[0] == 1
    assert label.iloc[1] == 1
    assert label.iloc[2] == 0
    assert pd.isna(label.iloc[-1])


def test_event_phases_include_onset_continuation_recovery() -> None:
    phases = classify_event_phases(pd.Series([0.0, 1.0, 1.0, 0.0, float("nan")]))

    assert phases.tolist() == [
        "calm",
        "onset",
        "continuation",
        "recovery",
        "insufficient_forward_window",
    ]


def test_build_event_labels_has_expected_columns() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30, tz="UTC")
    frame = pd.DataFrame(
        {
            "event_time": dates,
            "forecast_timestamp": dates + pd.offsets.BDay(1),
            "spx_price": [100, *range(99, 70, -1)],
            "realized_vol_20": [0.1] * 30,
            "vix_like": [20] * 30,
            "credit_spread_like": [80] * 30,
            "liquidity_stress_like": [0.2] * 30,
        }
    )

    labels = build_event_labels(frame, horizon=5, drawdown_threshold=0.05)

    assert "drawdown_5pct_h5" in labels.columns
    assert {"onset", "continuation"}.issubset(set(labels["event_phase"]))
