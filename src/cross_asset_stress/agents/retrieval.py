"""Point-in-time retrieval helpers."""

from __future__ import annotations

from datetime import datetime

from cross_asset_stress.agents.schemas import AgentContextDocument
from cross_asset_stress.data.schemas import DocumentRecord


def documents_as_of(
    documents: list[DocumentRecord],
    *,
    as_of_timestamp: datetime,
    limit: int = 10,
) -> list[AgentContextDocument]:
    """Return agent context documents available as of a forecast timestamp."""

    eligible = [
        document
        for document in documents
        if document.earliest_valid_prediction_timestamp <= as_of_timestamp
    ]
    eligible = sorted(eligible, key=lambda doc: doc.earliest_valid_prediction_timestamp, reverse=True)
    return [
        AgentContextDocument(
            evidence_id=document.document_id,
            title=document.title,
            text=document.text,
            publication_timestamp=document.publication_timestamp,
            document_hash=document.document_hash,
        )
        for document in eligible[:limit]
    ]

