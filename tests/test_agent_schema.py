from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from chronoswan.agents.forecaster import MockAgentForecaster
from chronoswan.agents.schemas import AgentContextDocument, AgentForecast


def test_agent_forecast_schema_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValidationError):
        AgentForecast(
            as_of_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            forecast_horizon_days=20,
            event_definition="Forward stress event",
            probability=1.2,
            confidence=0.5,
            risk_factors=[],
            counterarguments=[],
            evidence_ids=[],
            data_quality_flags=[],
            abstain=False,
        )


def test_agent_forecast_cites_only_available_evidence() -> None:
    forecast = AgentForecast(
        as_of_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        forecast_horizon_days=20,
        event_definition="Forward stress event",
        probability=0.2,
        confidence=0.5,
        risk_factors=[],
        counterarguments=[],
        evidence_ids=["doc-2"],
        data_quality_flags=[],
        abstain=False,
    )

    with pytest.raises(ValueError):
        forecast.validate_evidence_ids({"doc-1"})


def test_mock_agent_returns_valid_schema_with_context() -> None:
    agent = MockAgentForecaster()
    doc = AgentContextDocument(
        evidence_id="doc-1",
        title="Credit stress rises",
        text="Liquidity and volatility indicators moved higher.",
        publication_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        document_hash="abc",
    )

    forecast = agent.forecast(
        as_of_timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
        structured_features={"vix_like": 35.0, "skew_like": 140.0},
        documents=[doc],
    )

    assert forecast.probability > 0
    forecast.validate_evidence_ids({"doc-1"})

