"""Base classes for data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


class UnavailableDataSourceError(RuntimeError):
    """Raised when an adapter is intentionally not implemented or unavailable."""


@dataclass(frozen=True)
class DataRequest:
    """Minimal request passed to data adapters."""

    start: datetime | str
    end: datetime | str
    symbols: tuple[str, ...] = ("SPX",)
    fields: tuple[str, ...] = ()


class DataAdapter(ABC):
    """Interface shared by public, proprietary, and synthetic data adapters."""

    source_name: str

    @abstractmethod
    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Return a DataFrame using the point-in-time contract."""


class TodoProprietaryAdapter(DataAdapter):
    """Adapter placeholder that fails loudly rather than inventing vendor fields."""

    source_name = "todo"

    def __init__(self, source_name: str, note: str | None = None) -> None:
        self.source_name = source_name
        self.note = note or "Adapter has not been implemented."

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        raise UnavailableDataSourceError(
            f"{self.source_name} is a TODO adapter. {self.note} "
            "Do not fabricate proprietary fields; define them from vendor metadata first."
        )

