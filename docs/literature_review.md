# Literature Context and Benchmark Map

Cross-Asset Stress Monitor is an extension of the original cross-asset correlation question. The paper keeps the same core idea, but measures co-movement through two complementary channels: realized cross-asset futures-price moves and option-implied volatility/skew moves. The option-implied layer is useful because it reflects the price of uncertainty, convexity demand, and downside protection before the full realized move is observed.

## 1. Option-Implied Tail Risk

The options literature motivates volatility, skew, downside tail probability, variance risk premium, and out-of-the-money put-demand measures as priced signals of market stress. Cboe's VIX methodology defines a model-free 30-day expected-volatility measure from SPX option prices, while SKEW is designed to summarize option-implied tail-risk information. Bollerslev, Tauchen, and Zhou connect variance risk premia to expected stock returns, and Kelly and Jiang show that time-varying tail risk is related to asset prices and macro conditions.

Role in this paper:

- Use ATM implied volatility as the cleanest cross-asset stress level.
- Use ES risk reversals and butterflies as downside-skew and smile-convexity proxies.
- Test whether top-decile option-implied stress states lift future SPX drawdown event rates.

## 2. Cross-Asset Dependence in Stress States

Extreme-dependence work shows that correlations estimated in tail states can differ from unconditional correlations. Longin and Solnik find stronger correlation behavior in bear-market tails, while Forbes and Rigobon show why crisis-period correlation increases can be biased by volatility. The implication is direct: conditional correlation is a useful first benchmark, but it should not be interpreted as causal attribution by itself.

Role in this paper:

- Compare selected-maturity futures-return correlations with 30-day ATM implied-volatility shock correlations.
- Condition cross-asset correlation analysis on ES/SPX stress states.
- Treat the result as stress co-movement and regime mapping, not proof of a single macro driver.

## 3. Dynamic Dependence and Spillovers

Dynamic conditional correlation and spillover-network models are the formal benchmarks for time-varying dependence and directional transmission. Engle's DCC framework models changing correlations, while Diebold and Yilmaz use forecast-error variance decompositions to measure volatility spillovers across equities, bonds, FX, and commodities.

Role in this paper:

- Use static and event-conditioned correlations as the first empirical layer.
- Reserve DCC and spillover-network tests as the natural external-validity benchmark for causal driver language.

## 4. PCA Absorption and Stress Concentration

PCA-based systemic-risk work motivates monitoring how much cross-sectional variation is explained by a small number of common components. Kritzman, Li, Page, and Rigobon's absorption ratio is the closest benchmark for the paper's PCA concentration measure.

Role in this paper:

- Estimate rolling PCA absorption on cross-asset ATM implied-volatility changes.
- Interpret high absorption as factor concentration, not as a standalone drawdown forecast.
- Pair PCA with conditional correlations so the result remains economically interpretable.

## 5. Forecasting and Backtest Evaluation

The forecasting design follows a simple rare-event validation standard: compare every model to unconditional event rates, judge probability models by calibration and ranking metrics, and test strategies chronologically with shifted signals and transaction costs. VIX term structure remains the primary direct forecasting benchmark because it is a known, liquid, option-implied measure of equity volatility risk.

Role in this paper:

- Report conditional event-rate lift before strategy testing.
- Compare VIX term-structure, option-implied stress, and combined structured models in a 2020+ holdout.
- Evaluate the AR/IVM rule as a protection overlay, not as a full return-forecasting model.

## References

- Bollerslev, T., Tauchen, G., and Zhou, H. (2009). "Expected Stock Returns and Variance Risk Premia." Review of Financial Studies. https://doi.org/10.1093/rfs/hhp008
- Cboe. "Cboe Volatility Index Methodology." https://cdn.cboe.com/resources/indices/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf
- Cboe. "The Cboe SKEW Index." https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf
- Diebold, F. X., and Yilmaz, K. (2012). "Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers." International Journal of Forecasting. https://doi.org/10.1016/j.ijforecast.2011.02.006
- Engle, R. F. (2002). "Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models." Journal of Business and Economic Statistics. https://doi.org/10.1198/073500102288618487
- Forbes, K. J., and Rigobon, R. (2002). "No Contagion, Only Interdependence: Measuring Stock Market Comovements." Journal of Finance. https://doi.org/10.1111/0022-1082.00494
- Kelly, B., and Jiang, H. (2014). "Tail Risk and Asset Prices." Review of Financial Studies. https://doi.org/10.1093/rfs/hhu039
- Kritzman, M., Li, Y., Page, S., and Rigobon, R. (2011). "Principal Components as a Measure of Systemic Risk." Journal of Portfolio Management. https://doi.org/10.3905/jpm.2011.37.4.112
- Longin, F., and Solnik, B. (2001). "Extreme Correlation of International Equity Markets." Journal of Finance. https://doi.org/10.1111/0022-1082.00340
