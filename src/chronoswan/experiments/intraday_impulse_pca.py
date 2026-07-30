"""Event-conditioned intraday correlation and PCA research workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chronoswan.data.intraday import (
    build_es_impulse_events,
    build_intraday_price_panel,
    build_intraday_return_panel,
    estimate_intraday_bars_per_date,
    load_bloomberg_intraday_workbook,
    summarize_intraday_coverage,
)


LITERATURE_CONTEXT = [
    {
        "topic": "Extreme dependence",
        "anchor": "Longin and Solnik show that cross-market correlations are asymmetric in extremes.",
        "url": "https://doi.org/10.1111/0022-1082.00340",
        "implication": "The large-move subset is economically motivated; unconditional correlation is a weak benchmark.",
    },
    {
        "topic": "Contagion versus interdependence",
        "anchor": "Forbes and Rigobon warn that crisis-period correlations are biased by higher volatility.",
        "url": "https://www.nber.org/papers/w7267",
        "implication": "Conditional correlations should be interpreted with volatility-scaling caveats.",
    },
    {
        "topic": "Dynamic correlations",
        "anchor": "Engle's DCC framework models time-varying conditional correlations.",
        "url": "https://doi.org/10.1198/073500102288618487",
        "implication": "Rolling and event-conditioned dependence should be compared with dynamic-correlation baselines.",
    },
    {
        "topic": "Spillovers",
        "anchor": "Diebold and Yilmaz measure return/volatility spillovers through forecast-error variance decompositions.",
        "url": "https://doi.org/10.1016/j.ijforecast.2012.08.006",
        "implication": "A later version can test directional driver claims with spillover networks.",
    },
    {
        "topic": "Principal-component systemic risk",
        "anchor": "Kritzman, Li, Page, and Rigobon use PCA concentration as an absorption-ratio stress indicator.",
        "url": "https://doi.org/10.2469/faj.v67.n1.5",
        "implication": "PCA is not new; the novel angle is point-in-time ES-impulse conditioning and driver attribution.",
    },
]


def run_intraday_impulse_pca(
    input_path: str | Path = "data/17sheets.xlsx",
    *,
    output_dir: str | Path = "data/processed",
    reference_ticker: str = "ES1",
    impulse_quantile: float = 0.95,
    rolling_window_days: int = 20,
    max_components: int = 5,
) -> dict[str, Any]:
    """Run the intraday ES impulse research workflow from a local workbook."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    long_frame = load_bloomberg_intraday_workbook(input_path)
    coverage = summarize_intraday_coverage(long_frame)
    price_panel = build_intraday_price_panel(long_frame)
    return_panel = build_intraday_return_panel(long_frame)
    events = build_es_impulse_events(
        return_panel,
        reference_ticker=reference_ticker,
        rolling_window_days=rolling_window_days,
        impulse_quantile=impulse_quantile,
    )

    event_summary = summarize_impulse_events(events)
    conditional_correlations = build_conditional_correlation_table(
        return_panel,
        events,
        reference_ticker=reference_ticker,
    )
    pca_summary, pca_loadings = run_conditional_pca_suite(
        return_panel,
        events,
        reference_ticker=reference_ticker,
        max_components=max_components,
    )
    rolling_pca = build_rolling_pca_absorption_table(
        return_panel,
        events,
        reference_ticker=reference_ticker,
        rolling_window_days=rolling_window_days,
    )
    predictive_results, predictive_coefficients = run_predictive_impulse_screen(
        return_panel,
        price_panel,
        events,
        reference_ticker=reference_ticker,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    long_frame.to_parquet(output_dir / "intraday_ohlcv_long.parquet", index=False)
    return_panel.to_parquet(output_dir / "intraday_return_panel.parquet")
    events.reset_index().to_parquet(output_dir / "intraday_es_impulse_events.parquet", index=False)
    coverage.to_csv(output_dir / "intraday_coverage.csv", index=False)
    event_summary.to_csv(output_dir / "intraday_event_summary.csv", index=False)
    conditional_correlations.to_csv(
        output_dir / "intraday_conditional_correlations.csv",
        index=False,
    )
    pca_summary.to_csv(output_dir / "intraday_pca_summary.csv", index=False)
    pca_loadings.to_csv(output_dir / "intraday_pca_loadings.csv", index=False)
    rolling_pca.to_csv(output_dir / "intraday_rolling_pca_absorption.csv", index=False)
    predictive_results.to_csv(output_dir / "intraday_predictive_screen.csv", index=False)
    predictive_coefficients.to_csv(
        output_dir / "intraday_predictive_coefficients.csv",
        index=False,
    )
    metadata = {
        "input_path": str(input_path),
        "reference_ticker": reference_ticker,
        "impulse_quantile": impulse_quantile,
        "rolling_window_days": rolling_window_days,
        "max_components": max_components,
        "n_sheets": int(coverage["ticker"].nunique()),
        "n_return_timestamps": int(len(return_panel)),
        "timestamp_start": str(return_panel.index.min()),
        "timestamp_end": str(return_panel.index.max()),
        "timezone_note": (
            "Workbook timestamps are used as exported. Confirm Bloomberg terminal timezone "
            "before making timestamp-level claims."
        ),
    }
    (output_dir / "intraday_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "long_frame": long_frame,
        "coverage": coverage,
        "price_panel": price_panel,
        "return_panel": return_panel,
        "events": events,
        "event_summary": event_summary,
        "conditional_correlations": conditional_correlations,
        "pca_summary": pca_summary,
        "pca_loadings": pca_loadings,
        "rolling_pca": rolling_pca,
        "predictive_results": predictive_results,
        "predictive_coefficients": predictive_coefficients,
        "literature_context": pd.DataFrame(LITERATURE_CONTEXT),
        "output_dir": output_dir,
    }


def summarize_impulse_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize event definitions and realized ES-move magnitudes."""

    ready = events["threshold_ready"] & events["es_return_1h"].notna()
    rows = []
    for event_name in ("large_abs", "large_down", "large_up"):
        mask = ready & events[event_name]
        moves = events.loc[mask, "es_return_1h"]
        rows.append(
            {
                "event": event_name,
                "eligible_bars": int(ready.sum()),
                "event_count": int(mask.sum()),
                "event_rate": float(mask.sum() / ready.sum()) if ready.sum() else np.nan,
                "median_es_return_bp": float(moves.median() * 10_000) if len(moves) else np.nan,
                "mean_abs_es_return_bp": float(moves.abs().mean() * 10_000) if len(moves) else np.nan,
                "p10_es_return_bp": float(moves.quantile(0.10) * 10_000) if len(moves) else np.nan,
                "p90_es_return_bp": float(moves.quantile(0.90) * 10_000) if len(moves) else np.nan,
                "rolling_window_bars": int(events["rolling_window_bars"].dropna().iloc[0]),
                "estimated_bars_per_date": float(
                    events["estimated_bars_per_date"].dropna().iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_conditional_correlation_table(
    return_panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    reference_ticker: str = "ES1",
) -> pd.DataFrame:
    """Compare unconditional and ES-impulse conditional driver correlations."""

    full = return_panel.join(events[["large_abs", "large_down", "large_up", "threshold_ready"]])
    samples = {
        "all_bars": full[reference_ticker].notna(),
        "threshold_ready": full["threshold_ready"] & full[reference_ticker].notna(),
        "large_abs": full["large_abs"] & full[reference_ticker].notna(),
        "large_down": full["large_down"] & full[reference_ticker].notna(),
        "large_up": full["large_up"] & full[reference_ticker].notna(),
    }
    rows: list[dict[str, object]] = []
    unconditional = {}
    for ticker in return_panel.columns:
        if ticker == reference_ticker:
            continue
        base = full.loc[samples["threshold_ready"], [reference_ticker, ticker]].dropna()
        unconditional[ticker] = _safe_corr(base[reference_ticker], base[ticker])
    for sample_name, mask in samples.items():
        for ticker in return_panel.columns:
            if ticker == reference_ticker:
                continue
            sample = full.loc[mask, [reference_ticker, ticker]].dropna()
            corr = _safe_corr(sample[reference_ticker], sample[ticker])
            rows.append(
                {
                    "sample": sample_name,
                    "driver": ticker,
                    "n_obs": int(len(sample)),
                    "corr_with_es": corr,
                    "abs_corr_with_es": abs(corr) if pd.notna(corr) else np.nan,
                    "delta_vs_ready_corr": corr - unconditional[ticker]
                    if pd.notna(corr) and pd.notna(unconditional[ticker])
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["sample", "abs_corr_with_es", "n_obs"],
        ascending=[True, False, False],
    )


def run_conditional_pca_suite(
    return_panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    reference_ticker: str = "ES1",
    max_components: int = 5,
    min_observations: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PCA on all bars and ES impulse subsets."""

    full = return_panel.join(events[["large_abs", "large_down", "large_up", "threshold_ready"]])
    sample_masks = {
        "threshold_ready": full["threshold_ready"] & full[reference_ticker].notna(),
        "large_abs": full["large_abs"] & full[reference_ticker].notna(),
        "large_down": full["large_down"] & full[reference_ticker].notna(),
        "large_up": full["large_up"] & full[reference_ticker].notna(),
    }
    summaries: list[dict[str, object]] = []
    loadings: list[dict[str, object]] = []
    for sample_name, mask in sample_masks.items():
        sample = full.loc[mask, return_panel.columns].copy()
        summary_rows, loading_rows = _fit_one_conditional_pca(
            sample,
            sample_name=sample_name,
            reference_ticker=reference_ticker,
            max_components=max_components,
            min_observations=min_observations,
        )
        summaries.extend(summary_rows)
        loadings.extend(loading_rows)
    return pd.DataFrame(summaries), pd.DataFrame(loadings)


def build_rolling_pca_absorption_table(
    return_panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    reference_ticker: str = "ES1",
    rolling_window_days: int = 20,
    n_components: int = 3,
    min_rows: int = 40,
    step_bars: int | None = None,
) -> pd.DataFrame:
    """Estimate rolling PCA concentration for all bars and large ES impulses."""

    bars_per_date = estimate_intraday_bars_per_date(return_panel[reference_ticker])
    if not np.isfinite(bars_per_date) or bars_per_date <= 0:
        bars_per_date = 24.0
    window_bars = max(min_rows, int(round(rolling_window_days * bars_per_date)))
    step = step_bars or max(1, int(round(bars_per_date)))
    rows = []
    es_bar_index = events.index[events["es_return_1h"].notna()]
    analysis_panel = return_panel.reindex(es_bar_index)
    analysis_events = events.reindex(es_bar_index)
    driver_columns = [column for column in return_panel.columns if column != reference_ticker]
    for end_position in range(window_bars, len(analysis_panel), step):
        window = analysis_panel.iloc[end_position - window_bars : end_position]
        event_mask = analysis_events["large_abs"].iloc[end_position - window_bars : end_position]
        event_window = window.loc[event_mask]
        rows.append(
            {
                "timestamp": analysis_panel.index[end_position - 1],
                "window_bars": window_bars,
                "all_bar_rows": int(len(window)),
                "large_abs_rows": int(len(event_window)),
                "all_bar_absorption": _pca_absorption_ratio(
                    window[driver_columns],
                    n_components=n_components,
                    min_rows=min_rows,
                ),
                "large_abs_absorption": _pca_absorption_ratio(
                    event_window[driver_columns],
                    n_components=n_components,
                    min_rows=max(8, min_rows // 4),
                ),
            }
        )
    return pd.DataFrame(rows)


def run_predictive_impulse_screen(
    return_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    reference_ticker: str = "ES1",
    test_fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a small chronological logistic screen for next-bar large downside moves."""

    frame = build_predictive_feature_frame(
        return_panel,
        price_panel,
        events,
        reference_ticker=reference_ticker,
    )
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"target_next_large_down", "target_next_large_abs"}
    ]
    rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for target in ("target_next_large_down", "target_next_large_abs"):
        data = frame.dropna(subset=[target]).copy()
        split = int(round(len(data) * (1.0 - test_fraction)))
        train = data.iloc[:split]
        test = data.iloc[split:]
        target_feature_columns = [
            column
            for column in feature_columns
            if train[column].notna().sum() >= 20 and train[column].std(skipna=True) > 0
        ]
        if (
            len(train) < 100
            or len(test) < 30
            or len(target_feature_columns) < 2
            or train[target].nunique() < 2
            or test[target].nunique() < 2
        ):
            rows.append(
                {
                    "target": target,
                    "model": "not_fit",
                    "status": "insufficient_class_balance",
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "n_features": int(len(target_feature_columns)),
                    "train_event_rate": float(train[target].mean()) if len(train) else np.nan,
                    "test_event_rate": float(test[target].mean()) if len(test) else np.nan,
                }
            )
            continue
        base_probability = float(train[target].mean())
        base_probabilities = np.full(len(test), base_probability)
        rows.append(
            {
                "target": target,
                "model": "train_base_rate",
                "status": "fit",
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "n_features": 0,
                "train_event_rate": float(train[target].mean()),
                "test_event_rate": float(test[target].mean()),
                "brier_score": float(brier_score_loss(test[target], base_probabilities)),
                "roc_auc": 0.5,
                "average_precision": float(test[target].mean()),
                "mean_predicted_probability": base_probability,
            }
        )
        model_specs = {
            "logit_unweighted": None,
            "logit_balanced": "balanced",
        }
        for model_name, class_weight in model_specs.items():
            _fit_predictive_logit(
                train=train,
                test=test,
                target=target,
                feature_columns=target_feature_columns,
                model_name=model_name,
                class_weight=class_weight,
                rows=rows,
                coefficient_rows=coefficient_rows,
            )
    coefficients = pd.DataFrame(coefficient_rows)
    if not coefficients.empty:
        coefficients = coefficients.sort_values(
            ["target", "model", "abs_coefficient"],
            ascending=[True, True, False],
        )
    return pd.DataFrame(rows), coefficients


def _fit_predictive_logit(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    model_name: str,
    class_weight: str | None,
    rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
) -> None:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.5,
                    max_iter=2_000,
                    class_weight=class_weight,
                    random_state=7,
                ),
            ),
        ]
    )
    model.fit(train[feature_columns], train[target])
    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    rows.append(
        {
            "target": target,
            "model": model_name,
            "status": "fit",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "n_features": int(len(feature_columns)),
            "train_event_rate": float(train[target].mean()),
            "test_event_rate": float(test[target].mean()),
            "brier_score": float(brier_score_loss(test[target], probabilities)),
            "roc_auc": float(roc_auc_score(test[target], probabilities)),
            "average_precision": float(average_precision_score(test[target], probabilities)),
            "mean_predicted_probability": float(np.mean(probabilities)),
        }
    )
    coefficients = model.named_steps["logit"].coef_[0]
    for feature, coefficient in zip(feature_columns, coefficients, strict=False):
        coefficient_rows.append(
            {
                "target": target,
                "model": model_name,
                "feature": feature,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
            }
        )


def build_predictive_feature_frame(
    return_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    reference_ticker: str = "ES1",
) -> pd.DataFrame:
    """Construct known-at-bar-close features for forecasting the next ES bar."""

    frame = pd.DataFrame(index=return_panel.index)
    eligible_returns = [
        column
        for column in return_panel.columns
        if return_panel[column].notna().mean() >= 0.15
    ]
    for column in eligible_returns:
        frame[f"ret_{column}"] = return_panel[column]
    frame["es_realized_vol_20h"] = return_panel[reference_ticker].rolling(20, min_periods=10).std()
    frame["es_realized_vol_60h"] = return_panel[reference_ticker].rolling(60, min_periods=20).std()
    frame["es_abs_ret_20h"] = return_panel[reference_ticker].abs().rolling(20, min_periods=10).mean()
    if {"UX1", "UX2"}.issubset(price_panel.columns):
        frame["ux2_minus_ux1"] = price_panel["UX2"] - price_panel["UX1"]
        frame["ux2_over_ux1"] = price_panel["UX2"] / price_panel["UX1"] - 1.0
    if {"NQ1", reference_ticker}.issubset(return_panel.columns):
        frame["nq_minus_es_ret"] = return_panel["NQ1"] - return_panel[reference_ticker]
    if {"RTY1", reference_ticker}.issubset(return_panel.columns):
        frame["rty_minus_es_ret"] = return_panel["RTY1"] - return_panel[reference_ticker]
    es_event_sequence = events.loc[events["es_return_1h"].notna(), ["large_down", "large_abs"]]
    frame["target_next_large_down"] = es_event_sequence["large_down"].shift(-1).reindex(
        frame.index
    ).astype(float)
    frame["target_next_large_abs"] = es_event_sequence["large_abs"].shift(-1).reindex(
        frame.index
    ).astype(float)
    frame = frame.loc[events["threshold_ready"].fillna(False)]
    return frame.replace([np.inf, -np.inf], np.nan)


def _fit_one_conditional_pca(
    sample: pd.DataFrame,
    *,
    sample_name: str,
    reference_ticker: str,
    max_components: int,
    min_observations: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    drivers = [
        column
        for column in sample.columns
        if column != reference_ticker and sample[column].notna().sum() >= min_observations
    ]
    driver_matrix, drivers = _select_complete_pca_matrix(
        sample[drivers],
        min_rows=min_observations,
    )
    es_returns = sample.loc[driver_matrix.index, reference_ticker].to_numpy()
    if len(driver_matrix) < min_observations or len(drivers) < 2:
        return (
            [
                {
                    "sample": sample_name,
                    "component": np.nan,
                    "status": "insufficient_complete_rows",
                    "n_rows": int(len(driver_matrix)),
                    "n_drivers": int(len(drivers)),
                }
            ],
            [],
        )
    n_components = min(max_components, len(drivers), len(driver_matrix) - 1)
    scaler = StandardScaler()
    x = scaler.fit_transform(driver_matrix[drivers])
    pca = PCA(n_components=n_components, random_state=7)
    scores = pca.fit_transform(x)

    summary_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    for component_index in range(n_components):
        component = component_index + 1
        component_scores = scores[:, component_index]
        if np.std(component_scores) == 0 or np.std(es_returns) == 0:
            pc_es_corr = np.nan
        else:
            pc_es_corr = float(np.corrcoef(component_scores, es_returns)[0, 1])
        orientation = -1.0 if pc_es_corr < 0 else 1.0
        aligned_corr = abs(pc_es_corr)
        aligned_loadings = pca.components_[component_index] * orientation
        order = np.argsort(np.abs(aligned_loadings))[::-1]
        top_positive = [
            drivers[idx]
            for idx in order
            if aligned_loadings[idx] > 0
        ][:3]
        top_negative = [
            drivers[idx]
            for idx in order
            if aligned_loadings[idx] < 0
        ][:3]
        summary_rows.append(
            {
                "sample": sample_name,
                "component": component,
                "status": "fit",
                "n_rows": int(len(driver_matrix)),
                "n_drivers": int(len(drivers)),
                "explained_variance_ratio": float(pca.explained_variance_ratio_[component_index]),
                "cumulative_explained_variance": float(
                    pca.explained_variance_ratio_[:component].sum()
                ),
                "abs_corr_with_es": aligned_corr,
                "top_positive_loadings": ", ".join(top_positive),
                "top_negative_loadings": ", ".join(top_negative),
            }
        )
        for driver, loading in zip(drivers, aligned_loadings, strict=False):
            loading_rows.append(
                {
                    "sample": sample_name,
                    "component": component,
                    "driver": driver,
                    "loading_aligned_to_es": float(loading),
                    "abs_loading": float(abs(loading)),
                    "explained_variance_ratio": float(
                        pca.explained_variance_ratio_[component_index]
                    ),
                    "abs_corr_with_es": aligned_corr,
                }
            )
    return summary_rows, loading_rows


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3:
        return np.nan
    if sample.iloc[:, 0].std() == 0 or sample.iloc[:, 1].std() == 0:
        return np.nan
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1]))


def _pca_absorption_ratio(
    frame: pd.DataFrame,
    *,
    n_components: int,
    min_rows: int,
) -> float:
    eligible = [
        column
        for column in frame.columns
        if frame[column].notna().sum() >= min_rows and frame[column].std(skipna=True) > 0
    ]
    sample, eligible = _select_complete_pca_matrix(frame[eligible], min_rows=min_rows)
    if len(sample) < min_rows or len(eligible) < 2:
        return np.nan
    components = min(n_components, len(eligible), len(sample) - 1)
    x = StandardScaler().fit_transform(sample)
    return float(PCA(n_components=components, random_state=7).fit(x).explained_variance_ratio_.sum())


def _select_complete_pca_matrix(
    frame: pd.DataFrame,
    *,
    min_rows: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Select a complete PCA matrix from an irregular intraday panel."""

    columns = [
        column
        for column in frame.columns
        if frame[column].notna().sum() >= min_rows and frame[column].std(skipna=True) > 0
    ]
    columns = sorted(columns, key=lambda column: frame[column].notna().sum(), reverse=True)
    while len(columns) >= 2:
        sample = frame[columns].dropna()
        if len(sample) >= min_rows:
            return sample, columns
        sparsest = min(columns, key=lambda column: frame[column].notna().sum())
        columns.remove(sparsest)
    return pd.DataFrame(index=frame.index), []
