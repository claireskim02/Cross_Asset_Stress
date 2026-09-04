"""End-to-end synthetic demonstration runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cross_asset_stress.agents.audit import audit_agent_forecast
from cross_asset_stress.agents.forecaster import MockAgentForecaster
from cross_asset_stress.agents.prompts import SYSTEM_PROMPT
from cross_asset_stress.data.synthetic import CLEAN_FEATURES, LEAKED_FEATURES, generate_synthetic_market_frame
from cross_asset_stress.events.labels import build_event_labels
from cross_asset_stress.experiments.registry import (
    ExperimentRecord,
    current_git_commit,
    make_experiment_id,
    stable_config_hash,
)
from cross_asset_stress.models.calibration import calibration_slope_intercept
from cross_asset_stress.models.logistic import fit_predict_logistic
from cross_asset_stress.models.naive import HistoricalFrequencyModel
from cross_asset_stress.validation.leakage_checks import LeakageFinding, audit_feature_matrix
from cross_asset_stress.validation.metrics import evaluate_probabilities, expected_calibration_error
from cross_asset_stress.validation.splits import chronological_split, purged_time_series_split


DEFAULT_TARGET = "drawdown_5pct_h20"


def prepare_synthetic_dataset(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Generate a synthetic feature/label table."""

    cfg = config or {}
    synthetic_cfg = cfg.get("synthetic", {})
    target_cfg = cfg.get("target", {})
    market = generate_synthetic_market_frame(
        start=synthetic_cfg.get("start", "1995-01-02"),
        end=synthetic_cfg.get("end", "2025-12-31"),
        seed=int(synthetic_cfg.get("seed", cfg.get("project", {}).get("seed", 42))),
    )
    labels = build_event_labels(
        market,
        horizon=int(target_cfg.get("horizon_days", 20)),
        drawdown_threshold=float(target_cfg.get("drawdown_threshold", 0.05)),
        volatility_quantile=float(target_cfg.get("volatility_quantile", 0.95)),
        vix_threshold=float(target_cfg.get("vix_threshold", 35.0)),
    )
    label_cols = [column for column in labels.columns if column not in {"event_time", "forecast_timestamp"}]
    dataset = market.merge(labels[["event_time", "forecast_timestamp", *label_cols]], on=["event_time", "forecast_timestamp"])
    return dataset


def _evaluate_model(
    dataset: pd.DataFrame,
    *,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    random_state: int = 42,
) -> dict[str, float | str | bool]:
    train = dataset.iloc[train_index]
    test = dataset.iloc[test_index]
    result = fit_predict_logistic(
        train[feature_cols],
        train[target_col],
        test[feature_cols],
        random_state=random_state,
    )
    metrics = evaluate_probabilities(test[target_col], result.probabilities)
    metrics.update(calibration_slope_intercept(test[target_col], result.probabilities))
    metrics["expected_calibration_error"] = expected_calibration_error(
        test[target_col],
        result.probabilities,
    )
    return {
        "model": model_name,
        "used_fallback": result.used_fallback,
        **metrics,
    }


def _evaluate_naive(
    dataset: pd.DataFrame,
    *,
    train_index: np.ndarray,
    test_index: np.ndarray,
    target_col: str,
) -> dict[str, float | str | bool]:
    train = dataset.iloc[train_index]
    test = dataset.iloc[test_index]
    model = HistoricalFrequencyModel().fit(train[target_col])
    probabilities = model.predict_proba(len(test))
    metrics = evaluate_probabilities(test[target_col], probabilities)
    metrics["expected_calibration_error"] = expected_calibration_error(test[target_col], probabilities)
    return {"model": "historical_unconditional_frequency", "used_fallback": False, **metrics}


def run_synthetic_demo(
    config: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = "data/processed",
) -> dict[str, Any]:
    """Run the synthetic leakage and validation demonstration."""

    cfg = config or {}
    seed = int(cfg.get("project", {}).get("seed", 42))
    target_col = cfg.get("target", {}).get("name", DEFAULT_TARGET)
    dataset = prepare_synthetic_dataset(cfg)
    feature_cols_clean = CLEAN_FEATURES
    feature_cols_leaky = CLEAN_FEATURES + LEAKED_FEATURES
    modeling = dataset.dropna(subset=[target_col, *feature_cols_leaky]).reset_index(drop=True)

    if modeling[target_col].nunique() < 2:
        raise ValueError("Synthetic target has fewer than two classes; adjust generation parameters")

    random_train_idx, random_test_idx = train_test_split(
        np.arange(len(modeling)),
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=modeling[target_col].astype(int),
    )
    chronological = chronological_split(modeling, train_fraction=float(cfg.get("validation", {}).get("train_fraction", 0.70)))
    purged_splits = list(
        purged_time_series_split(
            modeling["forecast_timestamp"],
            n_splits=int(cfg.get("validation", {}).get("n_splits", 3)),
            horizon=int(cfg.get("validation", {}).get("purge_horizon_days", 20)),
            embargo=int(cfg.get("validation", {}).get("embargo_days", 5)),
        )
    )
    purged = purged_splits[-1]

    rows = [
        _evaluate_model(
            modeling,
            train_index=random_train_idx,
            test_index=random_test_idx,
            feature_cols=feature_cols_leaky,
            target_col=target_col,
            model_name="random_split_logistic_with_known_leaks_invalid",
            random_state=seed,
        ),
        _evaluate_naive(
            modeling,
            train_index=chronological.train_index,
            test_index=chronological.test_index,
            target_col=target_col,
        ),
        _evaluate_model(
            modeling,
            train_index=chronological.train_index,
            test_index=chronological.test_index,
            feature_cols=feature_cols_clean,
            target_col=target_col,
            model_name="chronological_logistic_clean",
            random_state=seed,
        ),
        _evaluate_model(
            modeling,
            train_index=purged.train_index,
            test_index=purged.test_index,
            feature_cols=feature_cols_clean,
            target_col=target_col,
            model_name="purged_embargo_logistic_clean",
            random_state=seed,
        ),
    ]
    metrics = pd.DataFrame(rows)

    leakage_findings = audit_feature_matrix(
        modeling,
        target_col=target_col,
        feature_cols=feature_cols_leaky,
    )

    last_row = modeling.iloc[-60]
    agent = MockAgentForecaster()
    forecast = agent.forecast(
        as_of_timestamp=last_row["forecast_timestamp"].to_pydatetime(),
        forecast_horizon_days=int(cfg.get("target", {}).get("horizon_days", 20)),  # type: ignore[arg-type]
        structured_features={feature: float(last_row[feature]) for feature in feature_cols_clean},
        documents=[],
    )
    agent_audit = audit_agent_forecast(
        forecast,
        context=[],
        prompt=SYSTEM_PROMPT,
        model_id=agent.model_id,
    )

    config_hash = stable_config_hash(cfg)
    record = ExperimentRecord(
        experiment_id=make_experiment_id("synthetic-demo", config_hash),
        created_at=pd.Timestamp.utcnow().to_pydatetime(),
        git_commit=current_git_commit(),
        config_hash=config_hash,
        feature_set=feature_cols_clean,
        label_definition=target_col,
        forecast_horizon_days=int(cfg.get("target", {}).get("horizon_days", 20)),
        train_start=str(modeling.iloc[chronological.train_index[0]]["forecast_timestamp"]),
        train_end=str(modeling.iloc[chronological.train_index[-1]]["forecast_timestamp"]),
        test_start=str(modeling.iloc[chronological.test_index[0]]["forecast_timestamp"]),
        test_end=str(modeling.iloc[chronological.test_index[-1]]["forecast_timestamp"]),
        purge_horizon_days=int(cfg.get("validation", {}).get("purge_horizon_days", 20)),
        embargo_days=int(cfg.get("validation", {}).get("embargo_days", 5)),
        model_name="synthetic_baseline_suite",
        metrics={
            f"{row['model']}_brier_score": float(row["brier_score"])
            for row in rows
            if isinstance(row.get("brier_score"), float)
        },
        contamination_flags=[
            finding.column or finding.check
            for finding in leakage_findings
            if finding.severity in {"critical", "high"}
        ],
    )

    output: dict[str, Any] = {
        "dataset": modeling,
        "metrics": metrics,
        "leakage_findings": leakage_findings,
        "agent_audit": agent_audit,
        "experiment_record": record,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "synthetic_demo_metrics.csv", index=False)
        pd.DataFrame([_finding_to_dict(finding) for finding in leakage_findings]).to_csv(
            out / "synthetic_leakage_findings.csv",
            index=False,
        )
        (out / "synthetic_experiment_record.json").write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        output["output_dir"] = out
    return output


def _finding_to_dict(finding: LeakageFinding) -> dict[str, str | None]:
    return {
        "severity": finding.severity,
        "check": finding.check,
        "column": finding.column,
        "message": finding.message,
    }

