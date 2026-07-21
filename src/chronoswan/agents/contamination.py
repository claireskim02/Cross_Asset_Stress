"""Contamination scoring for agent experiments."""

from __future__ import annotations

from datetime import datetime

from chronoswan.agents.schemas import AgentContextDocument


EVENT_NAME_TERMS = {
    "black monday",
    "global financial crisis",
    "covid crash",
    "eurozone crisis",
    "flash crash",
    "dot-com crash",
}


def score_context_contamination(
    documents: list[AgentContextDocument],
    *,
    as_of_timestamp: datetime,
) -> float:
    """Heuristic contamination score for supplied agent context."""

    if not documents:
        return 0.0
    score = 0.0
    for document in documents:
        if document.publication_timestamp > as_of_timestamp:
            score += 0.60
        lowered = f"{document.title}\n{document.text}".lower()
        if any(term in lowered for term in EVENT_NAME_TERMS):
            score += 0.20
        if "after the crash" in lowered or "subsequent returns" in lowered:
            score += 0.20
    return min(1.0, score / max(1, len(documents)))

