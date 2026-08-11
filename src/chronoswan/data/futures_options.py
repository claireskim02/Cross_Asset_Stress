"""Nasdaq Data Link futures-options implied-volatility pulls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DATATABLE_ENDPOINT = "https://data.nasdaq.com/api/v3/datatables/{table_code}"


@dataclass(frozen=True)
class FuturesOptionContract:
    """Research mapping for one futures-options implied-volatility root."""

    research_ticker: str
    exchange_code: str
    futures_code: str
    option_code: str
    asset_class: str
    role: str


AR_IVM_COLUMNS = [
    "date",
    "exchange_code",
    "futures_code",
    "option_code",
    "expiration",
    "futures",
    "atm",
    "rr25",
    "rr10",
    "fly25",
    "fly10",
    "days_expiration",
    "days_termination",
]


AR_FUTURES_OPTION_UNIVERSE: tuple[FuturesOptionContract, ...] = (
    FuturesOptionContract("BO", "CBT", "BO", "BO", "agriculture", "soybean_oil_options"),
    FuturesOptionContract("C", "CBT", "C", "C", "agriculture", "corn_options"),
    FuturesOptionContract("ES", "CME", "ES", "ES", "equity_index", "spx_options_on_futures"),
    FuturesOptionContract("NQ", "CME", "NQ", "NQ", "equity_index", "nasdaq_options_on_futures"),
    FuturesOptionContract(
        "RTY",
        "CME",
        "RTY",
        "RTY",
        "equity_index",
        "russell_options_on_futures",
    ),
    FuturesOptionContract("TU", "CBT", "TU", "TU", "rates", "two_year_note_options"),
    FuturesOptionContract("FV", "CBT", "FV", "FV", "rates", "five_year_note_options"),
    FuturesOptionContract("TY", "CBT", "TY", "TY", "rates", "ten_year_note_options"),
    FuturesOptionContract("TN", "CBT", "TN", "TN", "rates", "ultra_ten_year_note_options"),
    FuturesOptionContract("US", "CBT", "US", "US", "rates", "classic_bond_options"),
    FuturesOptionContract("UB", "CBT", "UB", "UB", "rates", "ultra_bond_options"),
    FuturesOptionContract("CL", "NYM", "CL", "CL", "energy", "wti_crude_options"),
    FuturesOptionContract("B", "ICE", "B", "B", "energy", "brent_crude_options"),
    FuturesOptionContract("HO", "NYM", "HO", "HO", "energy", "heating_oil_options"),
    FuturesOptionContract("NG", "NYM", "NG", "NG", "energy", "natural_gas_options"),
    FuturesOptionContract("RB", "NYM", "RB", "RB", "energy", "gasoline_options"),
    FuturesOptionContract("GC", "CMX", "GC", "GC", "metals", "gold_options"),
    FuturesOptionContract("HG", "CMX", "HG", "HG", "metals", "copper_options"),
    FuturesOptionContract("SI", "CMX", "SI", "SI", "metals", "silver_options"),
    FuturesOptionContract("DX", "ICE", "DX", "DX", "currency", "dollar_index_options"),
    FuturesOptionContract("AD", "CME", "AD", "AD", "currency", "australian_dollar_options"),
    FuturesOptionContract("BP", "CME", "BP", "BP", "currency", "british_pound_options"),
    FuturesOptionContract("CD", "CME", "CD", "CD", "currency", "canadian_dollar_options"),
    FuturesOptionContract("EC", "CME", "EC", "EC", "currency", "euro_fx_options"),
    FuturesOptionContract("JY", "CME", "JY", "JY", "currency", "yen_options"),
    FuturesOptionContract("SF", "CME", "SF", "SF", "currency", "swiss_franc_options"),
    FuturesOptionContract("S", "CBT", "S", "S", "agriculture", "soybean_options"),
    FuturesOptionContract("SM", "CBT", "SM", "SM", "agriculture", "soybean_meal_options"),
    FuturesOptionContract("W", "CBT", "W", "W", "agriculture", "wheat_options"),
)


def fetch_ar_ivm_for_contract(
    contract: FuturesOptionContract,
    *,
    api_key: str,
    start_date: str,
    end_date: str | None = None,
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    """Fetch AR/IVM rows for one contract from Nasdaq Data Link Tables API."""

    params: dict[str, object] = {
        "exchange_code": contract.exchange_code,
        "futures_code": contract.futures_code,
        "option_code": contract.option_code,
        "date.gte": start_date,
        "qopts.columns": ",".join(AR_IVM_COLUMNS),
    }
    if end_date is not None:
        params["date.lte"] = end_date
    records = fetch_datatable_records(
        "AR/IVM",
        params=params,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    frame = pd.DataFrame(records)
    if frame.empty:
        return _empty_ivm_frame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in [
        "futures",
        "atm",
        "rr25",
        "rr10",
        "fly25",
        "fly10",
        "days_expiration",
        "days_termination",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["research_ticker"] = contract.research_ticker
    frame["asset_class"] = contract.asset_class
    frame["role"] = contract.role
    return frame.sort_values(["research_ticker", "date", "days_expiration"]).reset_index(drop=True)


def fetch_ar_ivm_universe(
    *,
    api_key: str,
    start_date: str,
    end_date: str | None = None,
    contracts: Iterable[FuturesOptionContract] = AR_FUTURES_OPTION_UNIVERSE,
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    """Fetch AR/IVM rows for the research futures-options universe."""

    frames = [
        fetch_ar_ivm_for_contract(
            contract,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            timeout_seconds=timeout_seconds,
        )
        for contract in contracts
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_ivm_frame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["research_ticker", "date", "days_expiration"]
    )


def build_constant_tenor_ivm_features(
    ivm: pd.DataFrame,
    *,
    target_days: tuple[int, ...] = (30, 60, 90),
    max_distance_days: int = 20,
) -> pd.DataFrame:
    """Select nearest option expiries and pivot IVM fields into a daily feature panel."""

    if ivm.empty:
        return pd.DataFrame()
    fields = [
        field
        for field in ["futures", "atm", "rr25", "rr10", "fly25", "fly10"]
        if field in ivm.columns
    ]
    candidates = ivm.dropna(subset=["date", "research_ticker", "days_expiration"]).copy()
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    rows: list[pd.DataFrame] = []
    for tenor in target_days:
        tenor_frame = candidates.copy()
        tenor_frame["target_days"] = tenor
        tenor_frame["distance_days"] = (tenor_frame["days_expiration"] - tenor).abs()
        tenor_frame = tenor_frame.loc[tenor_frame["distance_days"] <= max_distance_days]
        if tenor_frame.empty:
            continue
        selected = (
            tenor_frame.sort_values(
                ["research_ticker", "date", "distance_days", "days_expiration"]
            )
            .groupby(["research_ticker", "date", "target_days"], as_index=False)
            .head(1)
        )
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    selected = pd.concat(rows, ignore_index=True)
    long = selected.melt(
        id_vars=["date", "research_ticker", "target_days"],
        value_vars=fields,
        var_name="feature",
        value_name="value",
    )
    long["column"] = (
        long["research_ticker"]
        + "_"
        + long["feature"]
        + "_"
        + long["target_days"].astype(str)
        + "d"
    )
    panel = long.pivot_table(index="date", columns="column", values="value", aggfunc="last")
    panel.columns.name = None
    return panel.sort_index()


def summarize_ivm_pull(ivm: pd.DataFrame) -> pd.DataFrame:
    """Build a compact coverage summary for AR/IVM data."""

    if ivm.empty:
        return pd.DataFrame(
            columns=[
                "research_ticker",
                "rows",
                "start",
                "end",
                "unique_dates",
                "min_dte",
                "median_dte",
                "max_dte",
            ]
        )
    rows = []
    for ticker, group in ivm.groupby("research_ticker", sort=True):
        rows.append(
            {
                "research_ticker": ticker,
                "rows": int(len(group)),
                "start": group["date"].min(),
                "end": group["date"].max(),
                "unique_dates": int(group["date"].nunique()),
                "min_dte": float(group["days_expiration"].min()),
                "median_dte": float(group["days_expiration"].median()),
                "max_dte": float(group["days_expiration"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("research_ticker").reset_index(drop=True)


def fetch_datatable_records(
    table_code: str,
    *,
    params: dict[str, object],
    api_key: str,
    timeout_seconds: float = 30.0,
) -> list[dict[str, object]]:
    """Fetch all pages for a Nasdaq Data Link datatable request."""

    cursor_id: str | None = None
    records: list[dict[str, object]] = []
    columns: list[str] | None = None
    while True:
        page_params = dict(params)
        page_params["api_key"] = api_key
        if cursor_id:
            page_params["qopts.cursor_id"] = cursor_id
        payload = _fetch_datatable_payload(
            table_code,
            params=page_params,
            timeout_seconds=timeout_seconds,
        )
        datatable = payload.get("datatable", {})
        columns = columns or [column["name"] for column in datatable.get("columns", [])]
        data = datatable.get("data") or []
        records.extend(dict(zip(columns, row, strict=False)) for row in data)
        cursor_id = payload.get("meta", {}).get("next_cursor_id")
        if not cursor_id:
            return records


def _fetch_datatable_payload(
    table_code: str,
    *,
    params: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    query = urlencode(params)
    url = f"{DATATABLE_ENDPOINT.format(table_code=table_code)}?{query}"
    request = Request(url, headers={"User-Agent": "ChronoSwan/0.1 futures options pull"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Nasdaq Data Link HTTP {exc.code}: {_sanitize_message(body)}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Nasdaq Data Link request failed: {exc}") from exc


def _sanitize_message(message: str, *, max_length: int = 240) -> str:
    return " ".join(message.split())[:max_length]


def _empty_ivm_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *AR_IVM_COLUMNS,
            "research_ticker",
            "asset_class",
            "role",
        ]
    )
