"""Text feature placeholders for point-in-time documents."""

from __future__ import annotations

from collections.abc import Iterable


STRESS_TERMS = {
    "stress",
    "liquidity",
    "volatility",
    "drawdown",
    "credit",
    "funding",
    "default",
    "recession",
}


def stress_term_count(text: str, vocabulary: Iterable[str] = STRESS_TERMS) -> int:
    """Count a small deterministic stress vocabulary in a document."""

    lower = text.lower()
    return sum(lower.count(term.lower()) for term in vocabulary)

