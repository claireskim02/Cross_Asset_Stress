"""Market-data adapter placeholders."""

from __future__ import annotations

from chronoswan.data.base import TodoProprietaryAdapter


class BloombergMarketAdapter(TodoProprietaryAdapter):
    """TODO adapter for Bloomberg market data."""

    def __init__(self) -> None:
        super().__init__(
            "bloomberg_market",
            "Wire this only after export fields, adjustment policy, and release timing are documented.",
        )


class FinaeonMarketAdapter(TodoProprietaryAdapter):
    """TODO adapter for Finaeon market data."""

    def __init__(self) -> None:
        super().__init__(
            "finaeon_market",
            "Wire this only after source timestamps and data entitlement boundaries are documented.",
        )

