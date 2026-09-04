from pathlib import Path

from cross_asset_stress.data.nasdaq_data_link import (
    NasdaqDatasetCandidate,
    _sanitize_message,
    load_nasdaq_data_link_api_key,
)


def test_load_nasdaq_data_link_api_key_reads_env_without_value_leak(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "OTHER_KEY=ignored",
                "NASDAQ_DATA_LINK_API_KEY='secret-value'",
            ]
        )
    )

    assert load_nasdaq_data_link_api_key(env) == "secret-value"


def test_sanitize_message_compacts_and_truncates() -> None:
    message = "alpha\n beta\t" + ("x" * 300)

    sanitized = _sanitize_message(message, max_length=20)

    assert sanitized == "alpha beta xxxxxxxxx"
    assert "\n" not in sanitized


def test_nasdaq_dataset_candidate_defaults_priority() -> None:
    candidate = NasdaqDatasetCandidate("ES1", "CHRIS/CME_ES1", "E-mini S&P")

    assert candidate.priority == "candidate"

