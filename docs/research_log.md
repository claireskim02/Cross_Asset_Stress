# Research Log

Status: active empirical draft. The current paper branch uses long-history cross-asset futures-options implied-volatility data as the main stress-regime layer, with daily SPX/VIX benchmarks and an intraday ES impulse case study.

## Research Position

The viable project is not to predict undefined extreme events. The viable project is to detect stress regimes with point-in-time market data, quantify whether those regimes lift forward drawdown risk, and test whether de-risking rules improve portfolio path behavior after signal delays and costs.

Current working question:

Can cross-asset option-implied volatility, downside skew, and PCA concentration identify equity stress regimes early enough to improve drawdown control relative to VIX term-structure and buy-and-hold benchmarks?

## Current Data Stack

The main empirical layer is the Nasdaq Data Link `AR/IVM` futures-options panel. The expanded local pull covers 29 roots across equity indices, rates, FX, energy, metals, and agriculture from 2007-current where available.

The daily benchmark layer is a manually sourced Bloomberg workbook with SPX total return, VIX term structure, rates, credit, FX, commodities, and related stress indicators.

The intraday case-study layer is the 140-day Bloomberg 60-minute workbook. It is used to demonstrate ES impulse attribution through conditional correlations and event-conditioned PCA, not to carry the full cross-regime claim.

No raw vendor data, parquet caches, executed notebooks, reports, credentials, or API keys are committed.

## Current Findings

The cross-asset implied-volatility layer has meaningful event-rate signal. Equity-index ATM implied volatility is the strongest standalone indicator in the current run; ES downside-skew pressure also raises forward SPX drawdown rates.

Post-2018 ES ATM implied-volatility shocks are tightly linked to NQ and RTY, and moderately linked to rates, yen, gold, FX, energy, and metals. Selected futures returns recover the classic risk-on/risk-off structure: equity beta clusters together while dollar strength, duration, and yen often sit on the defensive side of ES risk.

PCA absorption is useful as a concentration diagnostic and communication layer. It is not currently strong enough to be treated as a standalone drawdown predictor.

VIX term structure remains the harder direct forecasting benchmark in the 2020+ holdout. The AR/IVM layer is more convincing as a stress state variable and exposure overlay input.

The simple AR/IVM de-risking overlay improves holdout Sharpe and drawdown behavior versus SPX total return, while giving up little raw return in the baseline rule. Sample-split validation shows that parameter selection is fragile: the validation-selected rule does not dominate in the 2020+ test. A high-conviction sensitivity rule using the pre-2020 95th-percentile stress threshold is an interesting research lead, but it must be locked before further testing.

Agriculture improves the macro interpretation set, especially for supply and inflation regimes. In the current equity drawdown tests, agricultural implied volatility is not a universal SPX drawdown-risk indicator.

## Current Decisions

Use `AR/IVM` for the first serious cross-asset option-implied stress panel. `AR/IVS` may be added later if the paper needs fuller surface geometry.

Keep the daily VIX term-structure model as the main direct forecasting benchmark. The options layer must be judged as incremental information, not in isolation.

Keep PCA in the workflow, but describe it as dimensionality reduction and stress concentration. Do not use PCA loadings alone to make causal macro-driver claims.

Use the intraday ES impulse workflow as an attribution case study. The 140-day intraday history is too short for the paper's final regime validation.

Use mock and Ollama agent paths only after the structured market-data benchmark is frozen. LLMs should synthesize timestamp-valid context and audit driver narratives, not replace the market-data benchmark.

## Agent Notes

The default agent remains `MockAgentForecaster`. It makes no API calls and costs nothing.

The optional Ollama route is the local open-weight path for future experiments. It does not solve look-ahead or contamination risk by itself; it only avoids paid API dependency and makes local model choice reproducible.

Any valid agent result must store:

- exact model identifier;
- declared or inferred knowledge cutoff when available;
- full prompt hash;
- context document hashes;
- evidence IDs;
- contamination flags;
- abstention state.

## Benchmark Notes

Minimum benchmark ladder:

- unconditional event frequency;
- conditional event-rate lift;
- VIX and VIX term-structure logistic model;
- option-implied stress logistic model;
- combined structured model;
- de-risking overlay with shifted signals and transaction costs;
- PCA absorption diagnostic;
- intraday ES impulse conditional correlation benchmark;
- event-conditioned PCA attribution case study.

Future benchmark additions:

- dynamic conditional correlation;
- spillover-network models;
- absorption-ratio baselines;
- locked hedge-instrument tests;
- formal walk-forward threshold selection.

## Open Questions

Should the primary strategy hedge be cash, Treasury futures, put spreads, VIX futures, or dynamic beta reduction?

Which target should become primary: 5% forward drawdown within 20 trading days, 10% drawdown, volatility regime, or time-to-stress hazard?

Can the high-conviction AR/IVM threshold survive a locked validation design, or is the holdout sensitivity result partly sample-specific?

Does `AR/IVS` surface geometry add enough information beyond ATM, risk reversal, butterfly, and constant-tenor features to justify the additional dimensionality?

What timestamp-valid macro/news archive can support an LLM synthesis layer without leaking future crisis narratives?

## Near-Term Next Step

Freeze a small set of AR/IVM stress rules, add formal hedge-instrument tests, and compare PCA absorption against DCC or spillover-network benchmarks before adding an LLM narrative layer.
