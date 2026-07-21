"""Macro and vintage-data adapter placeholders."""

from __future__ import annotations

from chronoswan.data.base import TodoProprietaryAdapter


class MacroVintageAdapter(TodoProprietaryAdapter):
    """TODO adapter for macroeconomic vintages."""

    def __init__(self, source_name: str = "macro_vintage_vendor") -> None:
        super().__init__(
            source_name,
            "Use vintages and release calendars. Never backfill latest revisions into history.",
        )

