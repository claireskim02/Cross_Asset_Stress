"""Leakage audits for feature matrices and agent context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from chronoswan.data.point_in_time import FutureDataError, validate_no_future_availability


LEAKY_NAME_PATTERNS = (
    "future",
    "forward",
    "leaked",
    "target",
    "label",
    "outcome",
    "post_event",
)


@dataclass(frozen=True)
class LeakageFinding:
    """A leakage finding emitted by an audit."""

    severity: str
    check: str
    column: str | None
    message: str


def find_suspicious_feature_names(
    columns: Iterable[str],
    patterns: Iterable[str] = LEAKY_NAME_PATTERNS,
) -> list[LeakageFinding]:
    """Flag feature names that often indicate forward information."""

    findings: list[LeakageFinding] = []
    for column in columns:
        lowered = column.lower()
        for pattern in patterns:
            if pattern in lowered:
                findings.append(
                    LeakageFinding(
                        severity="high",
                        check="feature_name",
                        column=column,
                        message=f"Feature name contains leakage pattern '{pattern}'.",
                    )
                )
                break
    return findings


def find_high_target_correlations(
    frame: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: Iterable[str],
    threshold: float = 0.90,
) -> list[LeakageFinding]:
    """Flag near-perfect linear association with the target."""

    if target_col not in frame.columns:
        raise KeyError(f"Missing target column: {target_col}")
    findings: list[LeakageFinding] = []
    target = pd.to_numeric(frame[target_col], errors="coerce")
    for column in feature_cols:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = target.notna() & values.notna()
        if valid.sum() < 10 or target.loc[valid].nunique() < 2:
            continue
        corr = float(np.corrcoef(target.loc[valid], values.loc[valid])[0, 1])
        if np.isfinite(corr) and abs(corr) >= threshold:
            findings.append(
                LeakageFinding(
                    severity="high",
                    check="target_correlation",
                    column=column,
                    message=f"Absolute correlation with target is {abs(corr):.3f}.",
                )
            )
    return findings


def audit_feature_matrix(
    frame: pd.DataFrame,
    *,
    target_col: str | None = None,
    forecast_time_col: str = "forecast_timestamp",
    availability_cols: Iterable[str] | None = None,
    feature_cols: Iterable[str] | None = None,
) -> list[LeakageFinding]:
    """Run lightweight leakage checks on a feature matrix."""

    findings: list[LeakageFinding] = []
    selected_features = list(feature_cols or _default_feature_columns(frame))
    findings.extend(find_suspicious_feature_names(selected_features))

    if availability_cols:
        try:
            validate_no_future_availability(
                frame,
                forecast_time_col=forecast_time_col,
                availability_cols=availability_cols,
            )
        except FutureDataError as exc:
            findings.append(
                LeakageFinding(
                    severity="critical",
                    check="future_availability",
                    column=None,
                    message=str(exc),
                )
            )

    if target_col:
        findings.extend(
            find_high_target_correlations(
                frame,
                target_col=target_col,
                feature_cols=selected_features,
            )
        )
    return findings


def raise_on_critical_findings(findings: Iterable[LeakageFinding]) -> None:
    """Raise when critical leakage findings are present."""

    critical = [finding for finding in findings if finding.severity == "critical"]
    if critical:
        messages = "; ".join(finding.message for finding in critical)
        raise FutureDataError(messages)


def _default_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("event_",)
    excluded = {
        "forecast_timestamp",
        "observation_time",
        "release_time",
        "ingestion_time",
        "earliest_valid_prediction_timestamp",
        "source",
        "vintage",
        "event_phase",
        "event_id",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and not column.endswith("__available_at")
        and not column.startswith(excluded_prefixes)
    ]

