"""Options-data adapter placeholders."""

from __future__ import annotations

from cross_asset_stress.data.base import TodoProprietaryAdapter


class OptionsAdapter(TodoProprietaryAdapter):
    """TODO adapter for option surface, skew, and tail-probability data."""

    def __init__(self, source_name: str = "options_vendor") -> None:
        super().__init__(
            source_name,
            "Expected future fields include IV level, skew, term structure, and OTM put demand.",
        )

