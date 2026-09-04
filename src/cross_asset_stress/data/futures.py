"""Futures-data adapter placeholders."""

from __future__ import annotations

from cross_asset_stress.data.base import TodoProprietaryAdapter


class FuturesAdapter(TodoProprietaryAdapter):
    """TODO adapter for index futures, VIX futures, and positioning data."""

    def __init__(self, source_name: str = "futures_vendor") -> None:
        super().__init__(
            source_name,
            "Document contract rolls, open interest timing, and settlement availability before use.",
        )

