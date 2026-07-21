"""Experiment registry helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ExperimentRecord(BaseModel):
    """Minimal reproducibility record for one experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    created_at: datetime
    git_commit: str
    config_hash: str
    data_manifest_hash: str | None = None
    feature_set: list[str]
    label_definition: str
    forecast_horizon_days: int
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    purge_horizon_days: int | None = None
    embargo_days: int | None = None
    model_name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    contamination_flags: list[str] = Field(default_factory=list)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must contain a mapping")
    return data


def stable_config_hash(config: dict[str, Any]) -> str:
    """Hash a config dict."""

    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit() -> str:
    """Return HEAD commit or a clear placeholder before the first commit."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "uncommitted-initial-repo"


def make_experiment_id(prefix: str, config_hash: str) -> str:
    """Stable-ish ID with a timestamp and config prefix."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{config_hash[:8]}"

