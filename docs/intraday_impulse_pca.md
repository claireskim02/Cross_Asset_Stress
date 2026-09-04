# Intraday ES Impulse PCA Research Note

This note documents the working design for the intraday cross-asset extension. It is intended to be publishable without exposing the underlying Bloomberg workbook or derived proprietary tables.

## Research Question

Can the largest 60-minute moves in ES1 be decomposed into recurring cross-asset drivers, and does PCA add information beyond a simple event-conditioned correlation table?

The empirical target is not a generic "market PCA." The target is a point-in-time map of large ES impulses:

- define large ES moves by a shifted rolling threshold;
- compare all-bar correlations with correlations only during large ES moves;
- fit PCA only on event-conditioned return panels;
- inspect whether the dominant factor looks like equity beta, duration hedge, oil shock, USD shock, volatility shock, or broad deleveraging;
- test whether pre-event state variables weakly forecast the next large ES impulse.

## Data Pull

Current local workbook:

- path: `data/17sheets.xlsx`;
- source: manual Bloomberg intraday export;
- frequency: 60-minute OHLCV bars;
- window: approximately the Bloomberg desktop/API intraday limit;
- tickers: ES1, NQ1, RTY1, TY1, FV1, US1, CL1, CO1, GC1, HG1, DXY, EURUSD, USDJPY, VIX, UX1, UX2, HYG, LQD, TLT, GLD, USO.

Raw and generated data are not committed. The loader writes a normalized OHLCV parquet cache and derived research tables under `data/processed/`.

## Method

The event definition uses the absolute ES1 60-minute log return. At each timestamp, the large-move threshold is the 95th percentile of absolute ES1 returns in the prior rolling 20-day window, shifted by one bar so the current move cannot set its own threshold.

Main benchmark:

- correlation of each cross-asset driver with ES1 across all threshold-ready bars;
- correlation of each driver with ES1 only during large absolute moves;
- separate conditional correlations for large downside and upside moves.

PCA layer:

- fit PCA on standardized driver returns, excluding ES1;
- run separate PCA fits for all threshold-ready bars, large absolute moves, large downside moves, and large upside moves;
- align component signs so positive loadings correspond to positive ES co-movement;
- report explained variance, component correlation with ES, and top positive/negative loadings;
- estimate rolling PCA concentration through an absorption-ratio-style statistic.

Predictive screen:

- forecast whether the next ES1 bar is a large downside or large absolute impulse;
- use only known-at-bar-close lagged/current cross-asset returns and rolling ES state;
- train before the last 30 percent of timestamps and test chronologically;
- treat results as diagnostic because 140 days is too short for crisis-level inference.

## Literature Position

This angle has precedent in work on extreme dependence, contagion/interdependence, dynamic conditional correlation, spillover networks, and PCA concentration as a systemic-risk indicator. The useful perspective in this note is to apply those ideas directly to large ES impulses, where the question is not whether PCA exists as a technique, but whether event-conditioned co-movement gives a cleaner picture of the cross-asset basket behind important equity moves.

The intraday exercise focuses on:

- event-conditioned PCA at the ES 60-minute impulse level;
- direct comparison against simple conditional correlation;
- rolling point-in-time thresholding rather than ex-post event selection;
- driver-attribution tables designed for a discretionary macro/risk discussion.

Anchor references:

- Longin and Solnik, "Extreme Correlation of International Equity Markets": https://doi.org/10.1111/0022-1082.00340
- Forbes and Rigobon, "No Contagion, Only Interdependence": https://www.nber.org/papers/w7267
- Engle, "Dynamic Conditional Correlation": https://doi.org/10.1198/073500102288618487
- Diebold and Yilmaz, "Better to Give than to Receive": https://doi.org/10.1016/j.ijforecast.2011.02.006
- Kritzman, Li, Page, and Rigobon, "Principal Components as a Measure of Systemic Risk": https://doi.org/10.3905/jpm.2011.37.4.112

## Presentation Boundary

For a public GitHub Pages version, publish methodology, benchmark definitions, and non-proprietary summaries. Do not publish the Bloomberg workbook, exact data extracts, or executed notebooks containing vendor-derived tables unless licensing permits it.
