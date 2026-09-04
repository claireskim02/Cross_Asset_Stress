# Research Design

Cross-Asset Stress Monitor asks whether cross-asset derivatives-market stress measures improve equity drawdown-risk monitoring. The central object is a point-in-time stress layer built from futures-options implied volatility, skew, dispersion, PCA concentration, and benchmark equity-volatility indicators.

## Framing

The project studies predefined stress events, not undefined extreme events. A valid signal is attached to a timestamp, horizon, event definition, and information set available at that timestamp.

The practical use case is portfolio protection. The research tests whether noisy daily and intraday moves can be filtered into a smaller set of states where equity drawdown risk is higher and cross-asset co-movement is more concentrated.

## Hypotheses

H1: Cross-asset option-implied stress features lift forward SPX drawdown event rates relative to unconditional base rates.

H2: Option-implied co-movement adds a useful risk layer beyond realized futures-return correlations because it captures priced uncertainty, skew, and convexity demand.

H3: PCA concentration is useful as a stress-state diagnostic, but not sufficient for causal macro-driver attribution without dynamic dependence or spillover tests.

H4: A simple shifted-signal de-risking overlay can improve drawdown and volatility behavior relative to SPX buy-and-hold, even if VIX term structure remains the stronger direct probability benchmark.

## Panels

Panel 1 is the primary daily empirical study. It uses the longest consistent AR/IVM futures-options history available locally, with daily benchmark indicators from Bloomberg.

Panel 2 is the intraday ES impulse case study. It uses Bloomberg 60-minute bars to test the event-conditioned correlation and PCA workflow, but it is not treated as final cross-regime evidence because the available history is short.

Panel 3 is the future implementation layer. It requires clean roll-adjusted continuous-futures returns and investable hedge instruments before the overlay can be evaluated as a deployable process.

## Benchmarks

The benchmark ladder is:

- unconditional historical frequency;
- conditional event-rate lift;
- VIX and VIX term-structure logistic models;
- option-implied stress logistic model;
- combined structured model;
- de-risking overlay with shifted signals and transaction costs;
- PCA absorption diagnostic;
- event-conditioned intraday correlation and PCA case study;
- external comparison against DCC, spillover-network, or related dynamic dependence models.

## Deliverable Standard

The research page should make four points clearly:

- what data is used and what is excluded from the public repository;
- how the stress score is constructed without look-ahead;
- whether stress states raise forward drawdown event rates;
- whether the overlay improves the portfolio path after signal delays and transaction costs.
