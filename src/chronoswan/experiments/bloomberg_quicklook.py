"""Quick Bloomberg case-study and benchmark pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from chronoswan.data.bloomberg_manual import (
    build_bloomberg_quicklook_panel,
    load_bloomberg_wide_xlsx,
)
from chronoswan.models.calibration import calibration_slope_intercept
from chronoswan.models.logistic import fit_predict_logistic
from chronoswan.models.naive import HistoricalFrequencyModel
from chronoswan.validation.metrics import evaluate_probabilities, expected_calibration_error


BLOOMBERG_QUICK_FEATURES = [
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
]


CASE_STUDIES = {
    "2011 Euro/US downgrade stress": "2011-08-08",
    "2015 China/deval vol shock": "2015-08-24",
    "2018 Volmageddon": "2018-02-05",
    "2018 Q4 drawdown low": "2018-12-24",
    "2020 COVID crash peak VIX": "2020-03-16",
    "2022 inflation/rates stress": "2022-06-13",
    "2024 volatility shock": "2024-08-05",
}


def run_bloomberg_quicklook(
    input_path: str | Path = "data/raw/Book1.xlsx",
    *,
    output_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    """Run the first Bloomberg quicklook and write local derived outputs."""

    raw = load_bloomberg_wide_xlsx(input_path)
    panel = build_bloomberg_quicklook_panel(raw)
    case_studies = build_case_study_table(panel)
    model_results, top_risk_dates = run_holdout_benchmark(panel)
    base_rates = event_base_rates(panel)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "bloomberg_clean_daily_panel.parquet", index=False)
    case_studies.to_csv(out / "bloomberg_case_studies.csv", index=False)
    model_results.to_csv(out / "bloomberg_quick_model_results.csv", index=False)
    top_risk_dates.to_csv(out / "bloomberg_top_risk_dates.csv", index=False)
    base_rates.to_csv(out / "bloomberg_event_base_rates.csv", index=False)

    return {
        "raw": raw,
        "panel": panel,
        "case_studies": case_studies,
        "model_results": model_results,
        "top_risk_dates": top_risk_dates,
        "base_rates": base_rates,
        "output_dir": out,
    }


def event_base_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Return base rates for selected event definitions."""

    labels = [
        "event_dd5_h5",
        "event_dd10_h5",
        "event_dd5_h20",
        "event_dd10_h20",
        "event_dd15_h60",
        "event_dd20_h60",
    ]
    rows = [
        {
            "label": label,
            "base_rate": float(panel[label].mean()),
            "base_rate_pct": float(panel[label].mean() * 100),
        }
        for label in labels
        if label in panel
    ]
    return pd.DataFrame(rows)


def build_case_study_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a compact stress-event snapshot table."""

    rows: list[dict[str, object]] = []
    for name, date in CASE_STUDIES.items():
        idx = (panel["date"] - pd.Timestamp(date)).abs().idxmin()
        row = panel.loc[idx]
        prior20 = panel.loc[max(0, idx - 20), "SPX"]
        prior60 = panel.loc[max(0, idx - 60), "SPX"]
        rows.append(
            {
                "case": name,
                "date": row["date"].date().isoformat(),
                "SPX": row["SPX"],
                "VIX": row.get("VIX"),
                "VVIX": row.get("VVIX"),
                "SKEW": row.get("SKEW"),
                "VIX3M_minus_VIX": row.get("vix3m_minus_vix"),
                "UX2_minus_UX1": row.get("ux2_minus_ux1"),
                "RealizedVol20_pct": row.get("realized_vol_20"),
                "TrailingDD60_pct": row.get("drawdown_60") * 100,
                "Prior20_SPX_ret_pct": (row["SPX"] / prior20 - 1.0) * 100,
                "Prior60_SPX_ret_pct": (row["SPX"] / prior60 - 1.0) * 100,
                "Forward20_minDD_pct": row.get("fwd_drawdown_20d") * 100,
                "Forward60_minDD_pct": row.get("fwd_drawdown_60d") * 100,
            }
        )
    return pd.DataFrame(rows)


def run_holdout_benchmark(
    panel: pd.DataFrame,
    *,
    target: str = "event_dd5_h20",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a quick 2010-2019 train / 2020+ holdout benchmark."""

    model_frame = panel.dropna(subset=[target]).reset_index(drop=True)
    features = [
        feature
        for feature in BLOOMBERG_QUICK_FEATURES
        if feature in model_frame and model_frame[feature].notna().mean() > 0.80
    ]
    train_index = model_frame.index[model_frame["date"] < "2020-01-01"].to_numpy()
    test_index = model_frame.index[model_frame["date"] >= "2020-01-01"].to_numpy()
    train_index = train_index[:-20]

    rows: list[dict[str, object]] = []
    train = model_frame.iloc[train_index]
    test = model_frame.iloc[test_index].copy()

    base = HistoricalFrequencyModel().fit(train[target])
    base_prob = base.predict_proba(len(test))
    rows.append({"model": "historical_base_rate", **_score(test[target], base_prob)})

    logistic = fit_predict_logistic(train[features], train[target], test[features])
    logistic_prob = logistic.probabilities
    rows.append({"model": "class_weighted_logistic_structured", **_score(test[target], logistic_prob)})

    top_risk = test.assign(pred_prob_dd5_h20=logistic_prob).sort_values(
        "pred_prob_dd5_h20",
        ascending=False,
    )
    top_cols = [
        "date",
        "pred_prob_dd5_h20",
        target,
        "fwd_drawdown_20d",
        "SPX",
        "VIX",
        "VVIX",
        "SKEW",
        "vix3m_minus_vix",
        "realized_vol_20",
        "drawdown_60",
    ]
    return pd.DataFrame(rows), top_risk[top_cols].head(100)


def _score(y_true: pd.Series, y_prob) -> dict[str, float]:
    metrics = evaluate_probabilities(y_true, y_prob)
    metrics.update(calibration_slope_intercept(y_true, y_prob))
    metrics["expected_calibration_error"] = expected_calibration_error(y_true, y_prob)
    return metrics

