"""Minimal Finaeon API client.

The client intentionally implements authentication and generic POST plumbing only.
Endpoint-specific payloads should be added after their OpenAPI schemas are reviewed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FinaeonConfigError(RuntimeError):
    """Raised when local Finaeon configuration is incomplete."""


class FinaeonAPIError(RuntimeError):
    """Raised when Finaeon returns an error or malformed auth response."""


@dataclass(frozen=True)
class FinaeonCredentials:
    """Credentials and connection settings loaded from the local environment."""

    base_url: str
    username: str
    password: str
    bearer_token: str | None = None
    timeout_seconds: int = 30
    user_agent: str = "CrossAssetStressMonitor/0.1 research-client"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "FinaeonCredentials":
        """Load credentials from environment variables and an optional .env file."""

        if env_file is not None:
            load_env_file(env_file)

        base_url = os.environ.get(
            "FINAEON_BASE_URL",
            "https://api.finaeon.com",
        ).strip()
        username = (
            os.environ.get("FINAEON_USERNAME")
            or os.environ.get("FINAEON_USER_NAME")
            or ""
        ).strip()
        password = os.environ.get("FINAEON_PASSWORD", "").strip()
        bearer_token = os.environ.get("FINAEON_BEARER_TOKEN", "").strip() or None
        timeout_raw = os.environ.get("FINAEON_TIMEOUT_SECONDS", "30").strip()
        user_agent = os.environ.get("FINAEON_USER_AGENT", "CrossAssetStressMonitor/0.1 research-client").strip()

        missing = []
        if not base_url:
            missing.append("FINAEON_BASE_URL")
        if not bearer_token:
            if not username:
                missing.append("FINAEON_USERNAME")
            if not password:
                missing.append("FINAEON_PASSWORD")
        if missing:
            raise FinaeonConfigError(
                "Missing Finaeon configuration: "
                + ", ".join(missing)
                + ". Fill these in a local .env file, export them in your shell, "
                "or provide FINAEON_BEARER_TOKEN."
            )

        return cls(
            base_url=base_url.rstrip("/"),
            username=username,
            password=password,
            bearer_token=bearer_token,
            timeout_seconds=int(timeout_raw),
            user_agent=user_agent,
        )


@dataclass(frozen=True)
class FinaeonAuthResult:
    """Authentication response metadata."""

    status_code: int
    token: str
    token_source: str


class FinaeonClient:
    """Small HTTP client for Finaeon login and generic authenticated calls."""

    def __init__(self, credentials: FinaeonCredentials) -> None:
        self.credentials = credentials
        self._token: str | None = credentials.bearer_token

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "FinaeonClient":
        return cls(FinaeonCredentials.from_env(env_file=env_file))

    def login(self) -> FinaeonAuthResult:
        """Authenticate via POST /login and store the returned bearer token."""

        payload = {
            "userName": self.credentials.username,
            "password": self.credentials.password,
        }
        status_code, headers, body = self._request(
            "POST",
            "/login",
            payload=payload,
            token=None,
        )
        token, source = extract_bearer_token(body, headers)
        if not token:
            raise FinaeonAPIError(
                "Finaeon login succeeded but no bearer token was found in the response. "
                f"Token extraction source={source}. "
                "If this is html_response, the configured base URL is likely a browser-only proxy."
            )
        self._token = token
        return FinaeonAuthResult(status_code=status_code, token=token, token_source=source)

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | list[Any] | str | None:
        """POST JSON to an authenticated Finaeon endpoint."""

        token = self._token or self.login().token
        request_payload = dict(payload)
        request_payload.setdefault("token", token)
        _, _, body = self._request("POST", endpoint, payload=request_payload, token=None)
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    def search(
        self,
        search_string: str,
        *,
        search_type: str = "symbol",
        base_filter: str = "exactmatch",
        sort: str = "pop",
        page: int | None = 1,
        page_size: int | None = 25,
    ) -> dict[str, Any] | list[Any] | str | None:
        """Search for Finaeon series metadata."""

        payload: dict[str, Any] = {
            "searchString": search_string,
            "searchType": search_type,
            "baseFilter": base_filter,
            "sort": sort,
        }
        if page is not None:
            payload["page"] = str(page)
        if page_size is not None:
            payload["pageSize"] = str(page_size)
        return self.post("/search", payload)

    def series(
        self,
        series_name: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        periodicity: str = "Daily",
        split_adjusted: bool = True,
        close_only: bool = False,
        metadata: bool = True,
        point_in_time: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any] | str | None:
        """Request metadata and price data from POST /series."""

        payload: dict[str, Any] = {
            "seriesName": series_name,
            "periodicity": periodicity,
            "splitAdjusted": _bool_string(split_adjusted),
            "closeOnly": _bool_string(close_only),
            "metadata": _bool_string(metadata),
            "pointInTime": _bool_string(point_in_time),
        }
        if start_date:
            payload["startDate"] = start_date
        if end_date:
            payload["endDate"] = end_date
        if extra:
            payload.update(extra)
        return self.post("/series", payload)

    def legacy_get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> str:
        """Call a legacy GET endpoint using username/password query auth."""

        request_params = dict(params)
        request_params.setdefault("username", self.credentials.username)
        request_params.setdefault("password", self.credentials.password)
        query = urlencode(request_params)
        endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        separator = "&" if "?" in endpoint_path else "?"
        _, _, body = self._request(
            "GET",
            f"{endpoint_path}{separator}{query}",
            payload=None,
            token=None,
        )
        return body

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None,
        token: str | None,
    ) -> tuple[int, dict[str, str], str]:
        endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{self.credentials.base_url}{endpoint_path}"
        headers = {
            "Accept": "application/json, text/csv, text/plain, */*",
            "Content-Type": "application/json",
            "User-Agent": self.credentials.user_agent,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        request = Request(
            url,
            data=json.dumps(payload or {}).encode("utf-8") if method != "GET" else None,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.credentials.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return response.status, dict(response.headers), body
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise FinaeonAPIError(
                f"Finaeon {method} {endpoint_path} failed with HTTP {exc.code}: {body[:500]}"
            ) from exc
        except URLError as exc:
            raise FinaeonAPIError(
                f"Could not reach Finaeon at {self.credentials.base_url}: {exc.reason}"
            ) from exc


def extract_bearer_token(body: str, headers: dict[str, str]) -> tuple[str, str]:
    """Extract a bearer token from common JSON fields, raw text, or headers."""

    if _looks_like_html(body):
        return "", "html_response"

    auth_header = headers.get("Authorization") or headers.get("authorization")
    if auth_header:
        token = _strip_bearer(auth_header)
        return (token, "authorization_header") if _looks_like_token(token) else ("", "invalid_header")

    token_header = (
        headers.get("X-Auth-Token")
        or headers.get("x-auth-token")
        or headers.get("X-Access-Token")
        or headers.get("x-access-token")
    )
    if token_header:
        token = _strip_bearer(token_header)
        return (token, "token_header") if _looks_like_token(token) else ("", "invalid_header")

    stripped = body.strip()
    if not stripped:
        return "", "missing"

    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        token = _strip_bearer(stripped)
        return (token, "raw_body") if _looks_like_token(token) else ("", "invalid_raw_body")

    if isinstance(decoded, str):
        token = _strip_bearer(decoded)
        return (token, "json_string") if _looks_like_token(token) else ("", "invalid_json_string")

    if isinstance(decoded, dict):
        for key in (
            "token",
            "access_token",
            "accessToken",
            "bearerToken",
            "bearer_token",
            "jwt",
        ):
            value = decoded.get(key)
            if isinstance(value, str) and value.strip():
                token = _strip_bearer(value)
                return (token, f"json:{key}") if _looks_like_token(token) else ("", f"invalid_json:{key}")

    return "", "missing"


def _strip_bearer(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().startswith("bearer "):
        return stripped[7:].strip()
    return stripped


def _looks_like_html(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("<!doctype html") or lowered.startswith("<html")


def _looks_like_token(value: str) -> bool:
    stripped = value.strip()
    if not stripped or any(char.isspace() for char in stripped):
        return False
    if stripped.startswith("<") or stripped.endswith(">"):
        return False
    return len(stripped) >= 16


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from a local .env file."""

    env_path = Path(path)
    if not env_path.exists():
        raise FinaeonConfigError(f"Env file does not exist: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = value
