"""Strict schemas for LLM-agent forecasts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
ForecastHorizon = Literal[1, 5, 20, 60, 120]


class AgentContextDocument(BaseModel):
    """Document supplied to an agent."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    title: str
    text: str
    publication_timestamp: datetime
    document_hash: str


class AgentForecast(BaseModel):
    """Strict JSON-compatible forecast schema."""

    model_config = ConfigDict(extra="forbid")

    as_of_timestamp: datetime
    forecast_horizon_days: ForecastHorizon
    event_definition: str
    probability: Probability
    confidence: Probability
    risk_factors: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    abstain: bool = False

    def validate_evidence_ids(self, available_evidence_ids: set[str]) -> None:
        """Raise if the forecast cites evidence not supplied in context."""

        unsupported = sorted(set(self.evidence_ids) - available_evidence_ids)
        if unsupported:
            raise ValueError(f"Forecast cites unavailable evidence IDs: {unsupported}")


class AgentForecastAudit(BaseModel):
    """Metadata stored alongside strict agent output."""

    model_config = ConfigDict(extra="forbid")

    forecast: AgentForecast
    model_id: str
    context_hash: str
    prompt_hash: str
    contamination_risk: Probability
    retrieval_leakage_flag: bool
    parametric_memory_leakage_flag: bool

