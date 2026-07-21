"""Document stores for point-in-time agent context."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from chronoswan.data.schemas import DocumentRecord


class DocumentStore:
    """In-memory document store used until a real retrieval backend is added."""

    def __init__(self, documents: Iterable[DocumentRecord] = ()) -> None:
        self._documents = list(documents)

    def add(self, document: DocumentRecord) -> None:
        self._documents.append(document)

    def as_of(self, as_of_timestamp: datetime) -> list[DocumentRecord]:
        return [
            document
            for document in self._documents
            if document.earliest_valid_prediction_timestamp <= as_of_timestamp
        ]

