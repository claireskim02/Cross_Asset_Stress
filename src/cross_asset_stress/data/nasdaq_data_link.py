"""Nasdaq Data Link discovery and pull helpers.

The helpers in this module deliberately keep API keys out of returned records.
They are intended for entitlement/provenance checks before a dataset becomes
part of the research pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATA_LINK_DATASET_ENDPOINT = "https://data.nasdaq.com/api/v3/datasets/{code}/data.json"


@dataclass(frozen=True)
class NasdaqDatasetCandidate:
    """Candidate Nasdaq Data Link dataset code for one desired research series."""

    research_ticker: str
    dataset_code: str
    description: str
    priority: str = "candidate"


NASDAQ_DATA_LINK_CANDIDATES: tuple[NasdaqDatasetCandidate, ...] = (
    NasdaqDatasetCandidate("ES1", "CHRIS/CME_ES1", "E-mini S&P 500 continuous futures"),
    NasdaqDatasetCandidate("NQ1", "CHRIS/CME_NQ1", "E-mini Nasdaq-100 continuous futures"),
    NasdaqDatasetCandidate("RTY1", "CHRIS/CME_RTY1", "E-mini Russell 2000 continuous futures"),
    NasdaqDatasetCandidate("TY1", "CHRIS/CBOT_TY1", "10-year Treasury note continuous futures"),
    NasdaqDatasetCandidate("US1", "CHRIS/CBOT_US1", "30-year Treasury bond continuous futures"),
    NasdaqDatasetCandidate("FV1", "CHRIS/CBOT_FV1", "5-year Treasury note continuous futures"),
    NasdaqDatasetCandidate("CL1", "CHRIS/CME_CL1", "WTI crude oil continuous futures"),
    NasdaqDatasetCandidate("CO1", "CHRIS/ICE_B1", "Brent crude oil continuous futures"),
    NasdaqDatasetCandidate("GC1", "CHRIS/CME_GC1", "Gold continuous futures"),
    NasdaqDatasetCandidate("HG1", "CHRIS/CME_HG1", "Copper continuous futures"),
    NasdaqDatasetCandidate("UX1", "CHRIS/CBOE_VX1", "Front VIX continuous futures"),
    NasdaqDatasetCandidate("UX2", "CHRIS/CBOE_VX2", "Second VIX continuous futures"),
    NasdaqDatasetCandidate("SPY", "EOD/SPY", "SPDR S&P 500 ETF daily OHLCV"),
    NasdaqDatasetCandidate("HYG", "EOD/HYG", "High-yield credit ETF daily OHLCV"),
    NasdaqDatasetCandidate("LQD", "EOD/LQD", "Investment-grade credit ETF daily OHLCV"),
    NasdaqDatasetCandidate("TLT", "EOD/TLT", "Long-duration Treasury ETF daily OHLCV"),
    NasdaqDatasetCandidate("GLD", "EOD/GLD", "Gold ETF daily OHLCV"),
    NasdaqDatasetCandidate("USO", "EOD/USO", "Oil ETF daily OHLCV"),
    NasdaqDatasetCandidate("VIX", "FRED/VIXCLS", "Cboe VIX close through FRED mirror"),
)


def load_nasdaq_data_link_api_key(env_path: str | Path = ".env") -> str:
    """Read ``NASDAQ_DATA_LINK_API_KEY`` from an env file without printing it."""

    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "NASDAQ_DATA_LINK_API_KEY":
            cleaned = value.strip().strip('"').strip("'")
            if cleaned:
                return cleaned
    raise ValueError("NASDAQ_DATA_LINK_API_KEY is not set in env file")


def probe_nasdaq_data_link_candidate(
    candidate: NasdaqDatasetCandidate,
    *,
    api_key: str,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-10",
    timeout_seconds: float = 20.0,
) -> dict[str, object]:
    """Probe one candidate dataset and return sanitized status metadata."""

    params = urlencode(
        {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": api_key,
        }
    )
    url = f"{DATA_LINK_DATASET_ENDPOINT.format(code=candidate.dataset_code)}?{params}"
    request = Request(url, headers={"User-Agent": "CrossAssetStressMonitor/0.1 research audit"})
    base = {
        "research_ticker": candidate.research_ticker,
        "dataset_code": candidate.dataset_code,
        "description": candidate.description,
        "priority": candidate.priority,
    }
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            **base,
            "status": "http_error",
            "http_status": int(exc.code),
            "message": _sanitize_message(body),
            "rows": 0,
            "columns": "",
        }
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "error",
            "http_status": None,
            "message": _sanitize_message(str(exc)),
            "rows": 0,
            "columns": "",
        }

    dataset = payload.get("dataset_data", {})
    data = dataset.get("data") or []
    columns = dataset.get("column_names") or []
    return {
        **base,
        "status": "ok",
        "http_status": 200,
        "message": "",
        "rows": int(len(data)),
        "columns": ", ".join(map(str, columns)),
        "oldest_sample_date": data[-1][0] if data else "",
        "newest_sample_date": data[0][0] if data else "",
    }


def probe_nasdaq_data_link_candidates(
    candidates: Iterable[NasdaqDatasetCandidate] = NASDAQ_DATA_LINK_CANDIDATES,
    *,
    api_key: str,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-10",
    timeout_seconds: float = 20.0,
) -> list[dict[str, object]]:
    """Probe multiple dataset candidates and return sanitized records."""

    return [
        probe_nasdaq_data_link_candidate(
            candidate,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            timeout_seconds=timeout_seconds,
        )
        for candidate in candidates
    ]


def _sanitize_message(message: str, *, max_length: int = 240) -> str:
    compact = " ".join(message.split())
    return compact[:max_length]

