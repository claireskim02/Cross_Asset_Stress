"""Manual Bloomberg workbook ingestion utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_bloomberg_wide_xlsx(path: str | Path, *, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Load the manual Bloomberg workbook format used in the first research pull.

    Expected shape: tickers in row 1, Bloomberg formula descriptions in row 2,
    and observations beginning in row 3.
    """

    raw = pd.read_excel(path, sheet_name=sheet_name, header=0, skiprows=[1])
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    rename = {
        column: _clean_bloomberg_column(column)
        for column in raw.columns
        if column != "date"
    }
    frame = raw.rename(columns=rename).dropna(subset=["date"]).sort_values("date")
    frame = frame.reset_index(drop=True)
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_bloomberg_quicklook_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Build initial features and forward drawdown labels from the Bloomberg panel."""

    panel = frame.dropna(subset=["SPX", "VIX"]).copy().reset_index(drop=True)
    panel["forecast_timestamp"] = panel["date"] + pd.offsets.BDay(1)
    panel["spx_ret_1d"] = np.log(panel["SPX"]).diff()
    panel["spx_ret_5d"] = np.log(panel["SPX"]).diff(5)
    panel["spx_ret_20d"] = np.log(panel["SPX"]).diff(20)
    panel["realized_vol_20"] = panel["spx_ret_1d"].rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    panel["realized_vol_60"] = panel["spx_ret_1d"].rolling(60, min_periods=20).std() * np.sqrt(252) * 100
    panel["drawdown_60"] = panel["SPX"] / panel["SPX"].rolling(60, min_periods=20).max() - 1.0
    panel["drawdown_252"] = panel["SPX"] / panel["SPX"].rolling(252, min_periods=60).max() - 1.0

    optional_features = {
        "vix_change_5d": lambda x: x["VIX"].diff(5),
        "vix_change_20d": lambda x: x["VIX"].diff(20),
        "vix3m_minus_vix": lambda x: x["VIX3M"] - x["VIX"],
        "ux2_minus_ux1": lambda x: x["UX2"] - x["UX1"],
        "vvix_change_5d": lambda x: x["VVIX"].diff(5),
        "skew_change_20d": lambda x: x["SKEW"].diff(20),
        "es_basis_pct": lambda x: (x["ES1"] / x["SPX"] - 1.0) * 100,
        "yc_10y_2y": lambda x: x["USGG10YR"] - x["USGG2YR"],
        "yc_30y_3m": lambda x: x["USGG30YR"] - x["USGG3M"],
        "dxy_ret_20d": lambda x: np.log(x["DXY"]).diff(20),
    }
    for name, builder in optional_features.items():
        try:
            panel[name] = builder(panel)
        except KeyError:
            panel[name] = np.nan

    for horizon in (5, 20, 60):
        panel[f"fwd_drawdown_{horizon}d"] = _forward_min(panel["SPX"], horizon) / panel["SPX"] - 1.0
        for threshold in (0.05, 0.10, 0.15, 0.20):
            column = f"event_dd{int(threshold * 100)}_h{horizon}"
            panel[column] = (panel[f"fwd_drawdown_{horizon}d"] <= -threshold).astype(float)
            panel.loc[panel["SPX"].shift(-horizon).isna(), column] = np.nan
    return panel


def _clean_bloomberg_column(column: object) -> str:
    value = str(column)
    return (
        value.replace(" Index", "")
        .replace(" US Equity", "")
        .replace(" Curncy", "")
        .replace(" ", "_")
    )


def _forward_min(series: pd.Series, horizon: int) -> pd.Series:
    return pd.concat([series.shift(-step) for step in range(1, horizon + 1)], axis=1).min(axis=1)

