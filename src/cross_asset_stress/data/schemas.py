"""Pydantic schemas for point-in-time market records."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceType(str, Enum):
    """Broad source classes used in manifests and audit reports."""

    SYNTHETIC = "synthetic"
    PUBLIC = "public"
    PROPRIETARY = "proprietary"
    DERIVED = "derived"


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class AvailabilityMetadata(BaseModel):
    """Bitemporal metadata required for every feature-like observation."""

    model_config = ConfigDict(extra="forbid")

    event_time: datetime
    observation_time: datetime
    release_time: datetime
    ingestion_time: datetime
    earliest_valid_prediction_timestamp: datetime
    source: str
    source_type: SourceType = SourceType.DERIVED
    vintage: str | None = None
    transformation_window: str | None = None

    @model_validator(mode="after")
    def validate_availability_order(self) -> "AvailabilityMetadata":
        if self.ingestion_time < self.release_time:
            raise ValueError("ingestion_time must be >= release_time")
        if self.earliest_valid_prediction_timestamp < self.release_time:
            raise ValueError("earliest_valid_prediction_timestamp must be >= release_time")
        if self.earliest_valid_prediction_timestamp < self.ingestion_time:
            raise ValueError("earliest_valid_prediction_timestamp must be >= ingestion_time")
        return self


class FeatureRecord(AvailabilityMetadata):
    """A single long-form point-in-time feature observation."""

    symbol: str = "SPX"
    feature_name: str
    value: float | None
    is_known_leak: bool = False
    leak_reason: str | None = None

    @model_validator(mode="after")
    def validate_leak_reason(self) -> "FeatureRecord":
        if self.is_known_leak and not self.leak_reason:
            raise ValueError("known leakage features must include a leak_reason")
        return self


class DataManifest(BaseModel):
    """Small manifest for reproducible local data snapshots."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    created_at: datetime
    source: str
    source_type: SourceType
    row_count: int = Field(ge=0)
    content_hash: str
    description: str
    files: list[str] = Field(default_factory=list)
