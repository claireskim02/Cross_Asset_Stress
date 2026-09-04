from __future__ import annotations

import os

from cross_asset_stress.data.finaeon import FinaeonClient, FinaeonCredentials, extract_bearer_token, load_env_file


def test_extract_bearer_token_from_json_field() -> None:
    token, source = extract_bearer_token('{"accessToken":"Bearer abcdefghijklmnopqrstuvwxyz"}', {})

    assert token == "abcdefghijklmnopqrstuvwxyz"
    assert source == "json:accessToken"


def test_extract_bearer_token_from_header() -> None:
    token, source = extract_bearer_token("", {"Authorization": "Bearer header-token"})

    assert token == ""
    assert source == "invalid_header"


def test_extract_bearer_token_rejects_proxy_html() -> None:
    token, source = extract_bearer_token("<html><head><title>Shibboleth</title></head></html>", {})

    assert token == ""
    assert source == "html_response"


def test_extract_bearer_token_from_plausible_header() -> None:
    token, source = extract_bearer_token("", {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz"})

    assert token == "abcdefghijklmnopqrstuvwxyz"
    assert source == "authorization_header"


def test_credentials_load_from_env_file(tmp_path, monkeypatch) -> None:
    for key in [
        "FINAEON_BASE_URL",
        "FINAEON_USERNAME",
        "FINAEON_PASSWORD",
        "FINAEON_TIMEOUT_SECONDS",
    ]:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FINAEON_BASE_URL=https://example.test",
                "FINAEON_USERNAME=researcher",
                "FINAEON_PASSWORD=secret",
                "FINAEON_TIMEOUT_SECONDS=12",
            ]
        ),
        encoding="utf-8",
    )

    credentials = FinaeonCredentials.from_env(env_file=env_file)

    assert credentials.base_url == "https://example.test"
    assert credentials.username == "researcher"
    assert credentials.password == "secret"
    assert credentials.timeout_seconds == 12


def test_credentials_can_use_bearer_token_without_password(tmp_path, monkeypatch) -> None:
    for key in [
        "FINAEON_BASE_URL",
        "FINAEON_USERNAME",
        "FINAEON_PASSWORD",
        "FINAEON_BEARER_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FINAEON_BASE_URL=https://example.test",
                "FINAEON_BEARER_TOKEN=abcdefghijklmnopqrstuvwxyz",
            ]
        ),
        encoding="utf-8",
    )

    credentials = FinaeonCredentials.from_env(env_file=env_file)

    assert credentials.username == ""
    assert credentials.password == ""
    assert credentials.bearer_token == "abcdefghijklmnopqrstuvwxyz"


def test_env_file_overrides_empty_shell_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FINAEON_USERNAME", "")
    env_file = tmp_path / ".env"
    env_file.write_text("FINAEON_USERNAME=filled\n", encoding="utf-8")

    load_env_file(env_file)

    assert os.environ["FINAEON_USERNAME"] == "filled"


def test_authenticated_post_places_token_in_body(monkeypatch) -> None:
    credentials = FinaeonCredentials(
        base_url="https://example.test",
        username="user",
        password="password",
        bearer_token="abcdefghijklmnopqrstuvwxyz",
    )
    client = FinaeonClient(credentials)
    captured = {}

    def fake_request(method, endpoint, *, payload, token):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        captured["token"] = token
        return 200, {}, '{"ok": true}'

    monkeypatch.setattr(client, "_request", fake_request)

    response = client.post("/search", {"searchString": "AAPL"})

    assert response == {"ok": True}
    assert captured["payload"]["token"] == "abcdefghijklmnopqrstuvwxyz"
    assert captured["token"] is None


def test_series_payload_matches_modern_endpoint_style(monkeypatch) -> None:
    credentials = FinaeonCredentials(
        base_url="https://example.test",
        username="user",
        password="password",
        bearer_token="abcdefghijklmnopqrstuvwxyz",
    )
    client = FinaeonClient(credentials)
    captured = {}

    def fake_request(method, endpoint, *, payload, token):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return 200, {}, '{"rows": []}'

    monkeypatch.setattr(client, "_request", fake_request)

    client.series("MSFT", start_date="01/01/2019", end_date="12/31/2020", close_only=True)

    assert captured["endpoint"] == "/series"
    assert captured["payload"]["seriesName"] == "MSFT"
    assert captured["payload"]["startDate"] == "01/01/2019"
    assert captured["payload"]["closeOnly"] == "true"
    assert captured["payload"]["pointInTime"] == "true"
