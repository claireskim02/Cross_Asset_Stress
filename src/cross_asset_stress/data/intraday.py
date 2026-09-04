"""Intraday Bloomberg workbook ingestion and alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class IntradayInstrument:
    """Small metadata record for the manually pulled Bloomberg sheets."""

    ticker: str
    asset_class: str
    role: str


INSTRUMENTS: dict[str, IntradayInstrument] = {
    "ES1": IntradayInstrument("ES1", "equity_index_futures", "reference_risk_asset"),
    "NQ1": IntradayInstrument("NQ1", "equity_index_futures", "growth_beta"),
    "RTY1": IntradayInstrument("RTY1", "equity_index_futures", "small_cap_beta"),
    "TY1": IntradayInstrument("TY1", "rates_futures", "ten_year_treasury_duration"),
    "US1": IntradayInstrument("US1", "rates_futures", "long_bond_duration"),
    "FV1": IntradayInstrument("FV1", "rates_futures", "five_year_treasury_duration"),
    "CL1": IntradayInstrument("CL1", "energy_futures", "wti_crude"),
    "CO1": IntradayInstrument("CO1", "energy_futures", "brent_crude"),
    "GC1": IntradayInstrument("GC1", "metal_futures", "gold"),
    "HG1": IntradayInstrument("HG1", "metal_futures", "copper"),
    "DXY": IntradayInstrument("DXY", "currency", "broad_usd"),
    "EURUSD": IntradayInstrument("EURUSD", "currency", "euro_usd"),
    "USDJPY": IntradayInstrument("USDJPY", "currency", "dollar_yen"),
    "VIX": IntradayInstrument("VIX", "volatility_index", "spot_equity_volatility"),
    "UX1": IntradayInstrument("UX1", "volatility_futures", "front_vix_future"),
    "UX2": IntradayInstrument("UX2", "volatility_futures", "second_vix_future"),
    "HYG": IntradayInstrument("HYG", "credit_etf", "high_yield_credit"),
    "LQD": IntradayInstrument("LQD", "credit_etf", "investment_grade_credit"),
    "TLT": IntradayInstrument("TLT", "rates_etf", "long_duration_treasury_etf"),
    "GLD": IntradayInstrument("GLD", "commodity_etf", "gold_etf"),
    "USO": IntradayInstrument("USO", "commodity_etf", "oil_etf"),
}


def load_bloomberg_intraday_workbook(
    path: str | Path,
    *,
    sheet_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a multi-sheet Bloomberg intraday OHLCV workbook into long form.

    Each sheet is expected to contain timestamp, open, high, low, close, and
    volume columns, but the timestamp column is detected because manual
    Bloomberg/Excel exports can contain leading blank columns or rows.
    """

    path = Path(path)
    workbook = pd.ExcelFile(path)
    selected_sheets = list(sheet_names) if sheet_names is not None else workbook.sheet_names
    frames = []
    for sheet_name in selected_sheets:
        raw = workbook.parse(sheet_name=sheet_name, header=None)
        frames.append(parse_intraday_ohlcv_sheet(raw, sheet_name=sheet_name))
    if not frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "ticker",
                "sheet_name",
                "asset_class",
                "role",
                *OHLCV_COLUMNS,
            ]
        )
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["ticker", "timestamp"]).drop_duplicates(["ticker", "timestamp"])
    return data.reset_index(drop=True)


def parse_intraday_ohlcv_sheet(raw: pd.DataFrame, *, sheet_name: str) -> pd.DataFrame:
    """Parse one raw Excel sheet into normalized OHLCV rows."""

    timestamp_col, timestamps = _detect_timestamp_column(raw)
    if timestamp_col + 5 >= len(raw.columns):
        raise ValueError(f"{sheet_name}: timestamp column does not have five OHLCV columns after it")

    block = raw.iloc[:, timestamp_col : timestamp_col + 6].copy()
    block.columns = ["timestamp", *OHLCV_COLUMNS]
    block["timestamp"] = timestamps
    block = block.dropna(subset=["timestamp"])
    for column in OHLCV_COLUMNS:
        block[column] = pd.to_numeric(block[column], errors="coerce")
    block = block.dropna(subset=["close"])

    ticker = canonical_intraday_ticker(sheet_name)
    metadata = INSTRUMENTS.get(ticker, IntradayInstrument(ticker, "unknown", "unknown"))
    block["ticker"] = ticker
    block["sheet_name"] = sheet_name
    block["asset_class"] = metadata.asset_class
    block["role"] = metadata.role
    return block[
        ["timestamp", "ticker", "sheet_name", "asset_class", "role", *OHLCV_COLUMNS]
    ].sort_values("timestamp")


def canonical_intraday_ticker(sheet_name: str) -> str:
    """Convert Bloomberg sheet labels such as ``HYG US Equity`` to ``HYG``."""

    cleaned = " ".join(str(sheet_name).split())
    for suffix in (" US Equity", " Commodity", " Currency", " Index"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.replace(" ", "_")


def summarize_intraday_coverage(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a ticker-level coverage table for a long OHLCV frame."""

    rows: list[dict[str, object]] = []
    for ticker, group in long_frame.groupby("ticker", sort=True):
        timestamps = pd.to_datetime(group["timestamp"])
        per_day = timestamps.dt.date.value_counts()
        rows.append(
            {
                "ticker": ticker,
                "sheet_name": group["sheet_name"].iloc[0],
                "asset_class": group["asset_class"].iloc[0],
                "role": group["role"].iloc[0],
                "rows": int(len(group)),
                "start": timestamps.min(),
                "end": timestamps.max(),
                "unique_dates": int(timestamps.dt.date.nunique()),
                "median_bars_per_date": float(per_day.median()) if not per_day.empty else np.nan,
                "zero_volume_share": float((group["volume"].fillna(0) == 0).mean()),
                "missing_close_share": float(group["close"].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["asset_class", "ticker"]).reset_index(drop=True)


def build_intraday_price_panel(long_frame: pd.DataFrame, *, field: str = "close") -> pd.DataFrame:
    """Pivot long OHLCV rows into a timestamp-by-ticker price panel."""

    panel = long_frame.pivot_table(
        index="timestamp",
        columns="ticker",
        values=field,
        aggfunc="last",
    ).sort_index()
    panel.columns.name = None
    return panel


def build_intraday_return_panel(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Build native one-bar log returns for each ticker and align by timestamp."""

    data = long_frame.sort_values(["ticker", "timestamp"]).copy()
    data["return_1h"] = data.groupby("ticker", sort=False)["close"].transform(
        lambda close: np.log(close).diff()
    )
    panel = data.pivot_table(
        index="timestamp",
        columns="ticker",
        values="return_1h",
        aggfunc="last",
    ).sort_index()
    panel.columns.name = None
    return panel.dropna(how="all")


def estimate_intraday_bars_per_date(series: pd.Series) -> float:
    """Estimate the median number of non-missing bars per calendar date."""

    clean = series.dropna()
    if clean.empty:
        return np.nan
    dates = pd.Series(pd.to_datetime(clean.index).date, index=clean.index)
    per_date = dates.value_counts()
    return float(per_date.median())


def build_es_impulse_events(
    return_panel: pd.DataFrame,
    *,
    reference_ticker: str = "ES1",
    rolling_window_days: int = 20,
    impulse_quantile: float = 0.95,
    min_window_bars: int = 60,
) -> pd.DataFrame:
    """Flag large ES moves using a shifted rolling absolute-return threshold."""

    if reference_ticker not in return_panel:
        raise ValueError(f"Reference ticker {reference_ticker!r} not found in return panel")
    es_returns = return_panel[reference_ticker].dropna()
    bars_per_date = estimate_intraday_bars_per_date(es_returns)
    if not np.isfinite(bars_per_date) or bars_per_date <= 0:
        bars_per_date = 24.0
    rolling_window_bars = max(min_window_bars, int(round(rolling_window_days * bars_per_date)))
    min_periods = max(min_window_bars, int(round(rolling_window_bars * 0.35)))
    threshold = (
        es_returns.abs()
        .rolling(rolling_window_bars, min_periods=min_periods)
        .quantile(impulse_quantile)
        .shift(1)
    )
    events = pd.DataFrame(index=return_panel.index)
    events["es_return_1h"] = return_panel[reference_ticker]
    events["abs_es_return_1h"] = events["es_return_1h"].abs()
    events["rolling_abs_threshold"] = threshold.reindex(events.index)
    events["threshold_ready"] = events["rolling_abs_threshold"].notna()
    events["large_abs"] = (
        events["threshold_ready"]
        & events["es_return_1h"].notna()
        & (events["abs_es_return_1h"] >= events["rolling_abs_threshold"])
    )
    events["large_down"] = events["large_abs"] & (events["es_return_1h"] < 0)
    events["large_up"] = events["large_abs"] & (events["es_return_1h"] > 0)
    events["rolling_window_bars"] = rolling_window_bars
    events["estimated_bars_per_date"] = bars_per_date
    events.index.name = "timestamp"
    return events


def _detect_timestamp_column(raw: pd.DataFrame) -> tuple[int, pd.Series]:
    best_column = -1
    best_timestamps = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    best_count = 0
    for column in raw.columns:
        timestamps = _coerce_timestamp_like(raw[column])
        count = int(timestamps.notna().sum())
        if count > best_count:
            best_column = int(column)
            best_timestamps = timestamps
            best_count = count
    if best_column < 0 or best_count < 3:
        raise ValueError("Could not detect an intraday timestamp column")
    return best_column, best_timestamps


def _coerce_timestamp_like(series: pd.Series) -> pd.Series:
    mask = series.map(_looks_timestamp_like)
    parsed = pd.to_datetime(series.where(mask), errors="coerce")
    plausible = parsed.between(pd.Timestamp("1990-01-01"), pd.Timestamp("2100-01-01"))
    return parsed.where(plausible)


def _looks_timestamp_like(value: object) -> bool:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return stripped[0].isdigit() and any(separator in stripped for separator in ("-", "/", ":"))
