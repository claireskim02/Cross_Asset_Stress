"""Mock and future local-LLM agent forecasters."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from math import exp
from urllib.error import URLError
from urllib.request import Request, urlopen

from chronoswan.agents.prompts import SYSTEM_PROMPT
from chronoswan.agents.schemas import AgentContextDocument, AgentForecast, ForecastHorizon


class MockAgentForecaster:
    """Deterministic agent used to test schema, retrieval, and audit plumbing."""

    model_id = "mock-point-in-time-agent-v0"

    def forecast(
        self,
        *,
        as_of_timestamp: datetime,
        forecast_horizon_days: ForecastHorizon = 20,
        event_definition: str = "Forward 5 percent S&P 500 drawdown within horizon",
        structured_features: Mapping[str, float] | None = None,
        documents: Sequence[AgentContextDocument] = (),
    ) -> AgentForecast:
        """Return a schema-valid forecast without calling an external model."""

        features = structured_features or {}
        risk_score = 0.0
        risk_factors: list[str] = []

        if float(features.get("vix_like", 0.0)) > 30:
            risk_score += 1.0
            risk_factors.append("Elevated VIX-like volatility")
        if float(features.get("skew_like", 0.0)) > 130:
            risk_score += 0.8
            risk_factors.append("Elevated option-skew-like tail pricing")
        if float(features.get("credit_spread_like", 0.0)) > 160:
            risk_score += 0.7
            risk_factors.append("Credit-spread-like stress")
        if float(features.get("liquidity_stress_like", 0.0)) > 0.55:
            risk_score += 0.6
            risk_factors.append("Liquidity-stress-like conditions")

        cited_ids: list[str] = []
        for document in list(documents)[:3]:
            text = f"{document.title} {document.text}".lower()
            if any(term in text for term in ("stress", "liquidity", "volatility", "credit")):
                risk_score += 0.2
                cited_ids.append(document.evidence_id)

        probability = 1.0 / (1.0 + exp(-(risk_score - 2.0)))
        has_context = bool(features) or bool(documents)
        return AgentForecast(
            as_of_timestamp=as_of_timestamp,
            forecast_horizon_days=forecast_horizon_days,
            event_definition=event_definition,
            probability=float(probability if has_context else 0.0),
            confidence=float(min(0.85, 0.20 + 0.12 * len(risk_factors) + 0.05 * len(cited_ids))),
            risk_factors=risk_factors,
            counterarguments=[
                "Synthetic scaffold cannot establish real predictive validity.",
                "Market stress has a low base rate and overlapping crisis windows.",
            ],
            evidence_ids=cited_ids,
            data_quality_flags=[] if has_context else ["insufficient_context"],
            abstain=not has_context,
        )


class LocalLLMAgentForecaster:
    """TODO interface for local/open-weight model calls."""

    def __init__(self, model_id: str, base_url: str | None = None) -> None:
        self.model_id = model_id
        self.base_url = base_url

    def forecast(self, *_: object, **__: object) -> AgentForecast:
        raise NotImplementedError(
            "Local LLM calls are intentionally not wired in the first scaffold. "
            f"Use the strict schema and system prompt when adding {self.model_id}. "
            f"Prompt seed: {SYSTEM_PROMPT[:80]}..."
        )


class OllamaAgentForecaster:
    """Local open-weight forecaster using Ollama's HTTP API."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: int = 90,
    ) -> None:
        self.model_id = model_id or os.environ.get("CHRONOSWAN_LOCAL_MODEL", "llama3.1:8b")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def forecast(
        self,
        *,
        as_of_timestamp: datetime,
        forecast_horizon_days: ForecastHorizon = 20,
        event_definition: str = "Forward 5 percent S&P 500 drawdown within horizon",
        structured_features: Mapping[str, float] | None = None,
        documents: Sequence[AgentContextDocument] = (),
    ) -> AgentForecast:
        """Call a local Ollama model and validate the strict forecast schema."""

        prompt = self._build_prompt(
            as_of_timestamp=as_of_timestamp,
            forecast_horizon_days=forecast_horizon_days,
            event_definition=event_definition,
            structured_features=structured_features or {},
            documents=documents,
        )
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama locally or use provider='mock'. "
                f"Base URL: {self.base_url}"
            ) from exc

        raw_response = body.get("response", "")
        try:
            forecast = AgentForecast.model_validate_json(raw_response)
        except ValueError:
            forecast = AgentForecast.model_validate(json.loads(raw_response))
        forecast.validate_evidence_ids({document.evidence_id for document in documents})
        return forecast

    def _build_prompt(
        self,
        *,
        as_of_timestamp: datetime,
        forecast_horizon_days: ForecastHorizon,
        event_definition: str,
        structured_features: Mapping[str, float],
        documents: Sequence[AgentContextDocument],
    ) -> str:
        context = {
            "as_of_timestamp": as_of_timestamp.isoformat(),
            "forecast_horizon_days": forecast_horizon_days,
            "event_definition": event_definition,
            "structured_features": dict(structured_features),
            "documents": [document.model_dump(mode="json") for document in documents],
            "required_schema": {
                "as_of_timestamp": "ISO timestamp",
                "forecast_horizon_days": "one of 1, 5, 20, 60, 120",
                "event_definition": "string",
                "probability": "float between 0 and 1",
                "confidence": "float between 0 and 1",
                "risk_factors": "list of strings",
                "counterarguments": "list of strings",
                "evidence_ids": "list of supplied evidence_id values only",
                "data_quality_flags": "list of strings",
                "abstain": "boolean",
            },
        }
        return f"{SYSTEM_PROMPT}\n\nContext:\n{json.dumps(context, indent=2, sort_keys=True)}"
