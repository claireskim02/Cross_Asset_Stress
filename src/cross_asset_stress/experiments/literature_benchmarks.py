"""Literature-inspired structured benchmark replication.

This module evaluates canonical stress indicators against chronological holdouts.
It is intentionally modest: the goal is a defensible baseline map, not a final
trading model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross_asset_stress.data.bloomberg_manual import (
    build_bloomberg_quicklook_panel,
    load_bloomberg_wide_xlsx,
)
from cross_asset_stress.models.calibration import calibration_slope_intercept
from cross_asset_stress.models.logistic import fit_predict_logistic
from cross_asset_stress.models.naive import HistoricalFrequencyModel
from cross_asset_stress.validation.metrics import evaluate_probabilities, expected_calibration_error


FEATURE_GROUPS = {
    "base_rate": [],
    "spx_state": [
        "spx_ret_5d",
        "spx_ret_20d",
        "realized_vol_20",
        "drawdown_60",
        "drawdown_252",
    ],
    "vix_level": [
        "VIX",
        "vix_change_5d",
        "vix_change_20d",
    ],
    "vix_term_structure": [
        "VIX",
        "VIX3M",
        "vix3m_minus_vix",
        "UX1",
        "ux2_minus_ux1",
    ],
    "option_tail": [
        "VIX",
        "VVIX",
        "SKEW",
        "vvix_change_5d",
        "skew_change_20d",
    ],
    "rates_dxy": [
        "yc_10y_2y",
        "yc_30y_3m",
        "DXY",
        "dxy_ret_20d",
    ],
    "combined_structured": [
        "spx_ret_5d",
        "spx_ret_20d",
        "realized_vol_20",
        "drawdown_60",
        "drawdown_252",
        "VIX",
        "vix_change_5d",
        "vix_change_20d",
        "VIX3M",
        "VVIX",
        "SKEW",
        "vix3m_minus_vix",
        "UX1",
        "ux2_minus_ux1",
        "es_basis_pct",
        "yc_10y_2y",
        "yc_30y_3m",
        "DXY",
        "dxy_ret_20d",
    ],
}


LITERATURE_ANCHORS = [
    {
        "family": "VIX / implied volatility",
        "benchmark": "VIX, VIX changes",
        "interpretation": "Market-implied 30-day SPX volatility; strong concurrent stress signal.",
    },
    {
        "family": "Volatility term structure",
        "benchmark": "VIX3M - VIX, UX2 - UX1",
        "interpretation": "Inversion/backwardation indicates acute near-term stress.",
    },
    {
        "family": "Option tail risk",
        "benchmark": "SKEW, VVIX",
        "interpretation": "Skew/tail pricing and volatility-of-volatility; useful but not sufficient alone.",
    },
    {
        "family": "Rates and macro-financial conditions",
        "benchmark": "Yield-curve slopes, DXY",
        "interpretation": "Macro stress context; more relevant to recession horizon than crash timing.",
    },
]


def run_literature_benchmark_replication(
    input_path: str | Path = "data/raw/Book1.xlsx",
    *,
    output_dir: str | Path = "data/processed",
    holdout_start: str = "2020-01-01",
    purge_rows: int = 20,
) -> dict[str, Any]:
    """Run the literature benchmark replication on a manual Bloomberg workbook."""

    panel = build_bloomberg_quicklook_panel(load_bloomberg_wide_xlsx(input_path))
    conditional_rates = build_conditional_indicator_table(panel)
    model_results = run_feature_group_holdout(
        panel,
        holdout_start=holdout_start,
        purge_rows=purge_rows,
    )
    top_risk_dates = top_holdout_risk_dates(
        panel,
        holdout_start=holdout_start,
        purge_rows=purge_rows,
    )
    case_table = build_case_notes(panel)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    conditional_rates.to_csv(out / "literature_conditional_indicator_rates.csv", index=False)
    model_results.to_csv(out / "literature_feature_group_holdout.csv", index=False)
    top_risk_dates.to_csv(out / "literature_top_risk_dates.csv", index=False)
    case_table.to_csv(out / "literature_case_notes.csv", index=False)

    return {
        "panel": panel,
        "conditional_rates": conditional_rates,
        "model_results": model_results,
        "top_risk_dates": top_risk_dates,
        "case_table": case_table,
        "literature_anchors": pd.DataFrame(LITERATURE_ANCHORS),
        "output_dir": out,
    }


def build_conditional_indicator_table(
    panel: pd.DataFrame,
    *,
    targets: tuple[str, ...] = ("event_dd5_h20", "event_dd10_h20"),
) -> pd.DataFrame:
    """Estimate event-rate lift conditional on benchmark stress indicators."""

    indicator_masks = {
        "VIX > 30": panel["VIX"] > 30,
        "VIX top decile": panel["VIX"] >= panel["VIX"].quantile(0.90),
        "VIX3M - VIX < 0": panel["vix3m_minus_vix"] < 0,
        "UX2 - UX1 < 0": panel["ux2_minus_ux1"] < 0,
        "VVIX top decile": panel["VVIX"] >= panel["VVIX"].quantile(0.90),
        "SKEW top decile": panel["SKEW"] >= panel["SKEW"].quantile(0.90),
        "Trailing 60D DD < -5%": panel["drawdown_60"] <= -0.05,
        "Realized vol 20D top decile": panel["realized_vol_20"]
        >= panel["realized_vol_20"].quantile(0.90),
        "10Y-2Y inverted": panel["yc_10y_2y"] < 0,
    }

    rows: list[dict[str, object]] = []
    for target in targets:
        base_rate = float(panel[target].mean())
        for indicator, mask in indicator_masks.items():
            valid = panel[target].notna() & mask.notna()
            coverage = float((valid & mask).mean())
            conditional_rate = float(panel.loc[valid & mask, target].mean())
            rows.append(
                {
                    "target": target,
                    "indicator": indicator,
                    "sample_coverage": coverage,
                    "base_event_rate": base_rate,
                    "conditional_event_rate": conditional_rate,
                    "lift_vs_base": conditional_rate / base_rate if base_rate else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "lift_vs_base"], ascending=[True, False])


def run_feature_group_holdout(
    panel: pd.DataFrame,
    *,
    targets: tuple[str, ...] = ("event_dd5_h20", "event_dd10_h20"),
    holdout_start: str = "2020-01-01",
    purge_rows: int = 20,
) -> pd.DataFrame:
    """Train literature-inspired feature groups before ``holdout_start`` and test after."""

    rows: list[dict[str, object]] = []
    for target in targets:
        model_frame = panel.dropna(subset=[target]).reset_index(drop=True)
        train_index = model_frame.index[model_frame["date"] < holdout_start].to_numpy()
        test_index = model_frame.index[model_frame["date"] >= holdout_start].to_numpy()
        if purge_rows:
            train_index = train_index[:-purge_rows]

        train = model_frame.iloc[train_index]
        test = model_frame.iloc[test_index]
        for group_name, candidate_features in FEATURE_GROUPS.items():
            features = [
                feature
                for feature in candidate_features
                if feature in model_frame and model_frame[feature].notna().mean() > 0.80
            ]
            if group_name == "base_rate":
                probabilities = HistoricalFrequencyModel().fit(train[target]).predict_proba(len(test))
            else:
                result = fit_predict_logistic(
                    train[features],
                    train[target],
                    test[features],
                    class_weight=None,
                )
                probabilities = result.probabilities
            rows.append(
                {
                    "target": target,
                    "feature_group": group_name,
                    "n_features": len(features),
                    **_score_probabilities(test[target], probabilities),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "brier_score"])


def top_holdout_risk_dates(
    panel: pd.DataFrame,
    *,
    target: str = "event_dd5_h20",
    holdout_start: str = "2020-01-01",
    purge_rows: int = 20,
    n_rows: int = 25,
) -> pd.DataFrame:
    """Return top predicted holdout dates for the VIX term-structure benchmark."""

    model_frame = panel.dropna(subset=[target]).reset_index(drop=True)
    features = FEATURE_GROUPS["vix_term_structure"]
    train_index = model_frame.index[model_frame["date"] < holdout_start].to_numpy()
    test_index = model_frame.index[model_frame["date"] >= holdout_start].to_numpy()
    if purge_rows:
        train_index = train_index[:-purge_rows]
    train = model_frame.iloc[train_index]
    test = model_frame.iloc[test_index].copy()
    result = fit_predict_logistic(
        train[features],
        train[target],
        test[features],
        class_weight=None,
    )
    columns = [
        "date",
        "pred_prob_dd5_h20",
        target,
        "fwd_drawdown_20d",
        "SPX",
        "VIX",
        "VIX3M",
        "vix3m_minus_vix",
        "UX1",
        "ux2_minus_ux1",
        "VVIX",
        "SKEW",
    ]
    test["pred_prob_dd5_h20"] = result.probabilities
    return test.sort_values("pred_prob_dd5_h20", ascending=False)[columns].head(n_rows)


def build_case_notes(panel: pd.DataFrame) -> pd.DataFrame:
    """Build compact notes for known stress windows."""

    cases = {
        "2011 US downgrade / euro stress": "2011-08-08",
        "2015 China devaluation / vol shock": "2015-08-24",
        "2018 Volmageddon": "2018-02-05",
        "2018 Q4 drawdown low": "2018-12-24",
        "COVID early-warning date": "2019-12-13",
        "2020 COVID peak VIX": "2020-03-16",
        "2022 inflation/rates stress": "2022-06-13",
        "2024 volatility shock": "2024-08-05",
    }
    rows: list[dict[str, object]] = []
    for name, date in cases.items():
        idx = (panel["date"] - pd.Timestamp(date)).abs().idxmin()
        row = panel.loc[idx]
        rows.append(
            {
                "case": name,
                "date": row["date"].date().isoformat(),
                "VIX": row["VIX"],
                "VVIX": row["VVIX"],
                "SKEW": row["SKEW"],
                "VIX3M_minus_VIX": row["vix3m_minus_vix"],
                "UX2_minus_UX1": row["ux2_minus_ux1"],
                "realized_vol_20": row["realized_vol_20"],
                "trailing_drawdown_60": row["drawdown_60"],
                "forward_drawdown_20": row["fwd_drawdown_20d"],
                "forward_drawdown_60": row["fwd_drawdown_60d"],
            }
        )
    return pd.DataFrame(rows)


def _score_probabilities(y_true: pd.Series, y_prob) -> dict[str, float]:
    metrics = evaluate_probabilities(y_true, y_prob)
    metrics.update(calibration_slope_intercept(y_true, y_prob))
    metrics["expected_calibration_error"] = expected_calibration_error(y_true, y_prob)
    return metrics
