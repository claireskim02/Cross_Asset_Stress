"""Agent forecast audits."""

from __future__ import annotations

import hashlib
import json

from chronoswan.agents.contamination import score_context_contamination
from chronoswan.agents.schemas import AgentContextDocument, AgentForecast, AgentForecastAudit


def stable_hash(payload: object) -> str:
    """Hash JSON-like payloads with stable key ordering."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_agent_forecast(
    forecast: AgentForecast,
    *,
    context: list[AgentContextDocument],
    prompt: str,
    model_id: str,
) -> AgentForecastAudit:
    """Validate evidence IDs and store contamination metadata."""

    forecast.validate_evidence_ids({document.evidence_id for document in context})
    contamination_risk = score_context_contamination(
        context,
        as_of_timestamp=forecast.as_of_timestamp,
    )
    return AgentForecastAudit(
        forecast=forecast,
        model_id=model_id,
        context_hash=stable_hash([document.model_dump() for document in context]),
        prompt_hash=stable_hash({"prompt": prompt}),
        contamination_risk=contamination_risk,
        retrieval_leakage_flag=contamination_risk > 0,
        parametric_memory_leakage_flag=False,
    )

