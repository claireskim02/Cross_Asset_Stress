# Literature Context And Benchmark Map

This is a working literature note, not a complete review. Its purpose is to map prior work to benchmark requirements so ChronoSwan does not mistake a useful engineering result for a novel empirical contribution.

## 1. Option-Implied Tail Risk

The options literature motivates volatility, skew, downside tail probability, variance risk premium, and OTM put-demand features as priced measures of uncertainty. Cboe's SKEW methodology is a direct example of extracting tail-risk information from S&P 500 option prices. Kelly and Jiang estimate time-varying tail risk from the cross section of stock returns and connect tail risk to asset prices.

Implication for ChronoSwan:

The structured benchmark must include derivatives-implied risk variables. A text or LLM agent must improve on these, not merely restate that markets are stressed when VIX or skew is high.

Current benchmark features:

- VIX level and change;
- VIX term structure or curve slope;
- cross-asset ATM implied volatility;
- risk reversals and downside-skew pressure;
- butterflies as smile-convexity proxies;
- option-implied PCA absorption;
- group-level stress features for equity indices, rates, FX, energy, metals, and agriculture.

Sources:

- Cboe SKEW white paper: https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf
- Kelly and Jiang, "Tail Risk and Asset Prices": https://www.nber.org/papers/w19375

## 2. Extreme Dependence And Contagion

Extreme-dependence work shows that correlations estimated in tail states can differ from unconditional correlations. Contagion work also warns that crisis-period correlations are mechanically affected by volatility, so a higher conditional correlation is not automatically a structural driver.

Implication for ChronoSwan:

Conditional correlation is the first benchmark for the ES impulse question. PCA must clarify concentration or factor composition beyond that simpler table. Driver language should remain cautious unless a formal lead-lag, spillover, or causal identification design is added.

Current implementation:

- 60-minute ES impulse events use a shifted rolling threshold;
- large-up, large-down, and large-absolute subsets are compared with threshold-ready observations;
- PCA is reported beside conditional correlations, not instead of them.

Sources:

- Longin and Solnik, "Extreme Correlation of International Equity Markets": https://doi.org/10.1111/0022-1082.00340
- Forbes and Rigobon, "No Contagion, Only Interdependence": https://www.nber.org/papers/w7267

## 3. Dynamic Correlation And Spillovers

Dynamic conditional correlation and spillover-network models are formal baselines for time-varying dependence. They are stronger benchmarks than a static correlation table when the research question becomes directional risk transmission.

Implication for ChronoSwan:

The current paper can claim that stress states are associated with changing cross-asset dependence. It should not yet claim directional causality. A later version should compare PCA concentration against DCC or Diebold-Yilmaz connectedness measures.

Sources:

- Engle, "Dynamic Conditional Correlation": https://doi.org/10.1198/073500102288618487
- Diebold and Yilmaz, "Better to Give than to Receive": https://doi.org/10.1016/j.ijforecast.2011.02.006

## 4. PCA Absorption And Systemic Risk

PCA-based systemic-risk work motivates monitoring the share of variance explained by a small number of components. A high absorption ratio can indicate that markets are moving under a more common factor, which is a useful stress-concentration concept.

Implication for ChronoSwan:

PCA is not the novel object by itself. The useful object is the point-in-time cross-asset stress workflow: build option-implied features, estimate concentration, test event-rate lift, and evaluate an exposure overlay with shifted signals.

Current implementation:

- rolling first-three-PC absorption for 30d ATM implied-volatility changes;
- rolling first-three-PC absorption for selected-maturity futures returns;
- intraday event-conditioned PC1 summaries for large ES impulse subsets.

Source:

- Kritzman, Li, Page, and Rigobon, "Principal Components as a Measure of Systemic Risk": https://doi.org/10.3905/jpm.2011.37.4.112

## 5. Stress-Regime Forecasting

Crash or stress forecasting with logistic models, regime models, tree models, and boosted classifiers is not novel by itself. The relevant standard is calibrated probability forecasting under time-series validation, not raw accuracy or a backtest tuned after the fact.

Implication for ChronoSwan:

The benchmark must remain chronological and hard to beat. VIX term structure is the main direct forecasting hurdle. The option-implied layer should be evaluated both as a probability input and as a risk-control state variable.

Current metrics:

- Brier score;
- log loss;
- average precision;
- ROC AUC;
- conditional event-rate lift;
- cumulative return;
- annualized return and volatility;
- zero-risk-free-rate Sharpe;
- max drawdown;
- worst single day;
- turnover and mean exposure.

## 6. LLM Look-Ahead Bias

LLM finance experiments are highly sensitive to look-ahead and contamination. Glasserman and Lin study bias in GPT-based sentiment backtests and show why model knowledge and text identity can distort historical evaluation.

Implication for ChronoSwan:

The agent layer should be added only after the market-data benchmark is locked. Its job should be timestamp-valid synthesis and evidence auditing, not unconstrained prediction.

Agent records must store:

- exact model identifier;
- declared or inferred knowledge cutoff when available;
- full prompt hash;
- context document hashes;
- evidence IDs;
- contamination flags;
- abstention state.

Source:

- Glasserman and Lin, "Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis": https://arxiv.org/abs/2309.17322

## 7. Current Contribution Statement

The current ChronoSwan contribution is a disciplined research pipeline:

- build a long-history cross-asset futures-options stress panel;
- benchmark it against VIX term structure and SPX total return;
- use conditional event-rate lift before strategy claims;
- use PCA as concentration measurement rather than causal proof;
- preserve an intraday ES impulse workflow for attribution;
- reserve agentic synthesis for a later contamination-controlled stage.

The strongest current empirical claim is that cross-asset option-implied stress is useful for regime measurement and drawdown control. The paper should not yet claim a general rare-event forecasting breakthrough.
