# Research Design

ChronoSwan asks whether cross-asset derivatives-market stress measures can improve equity drawdown-risk monitoring, and whether a leakage-controlled LLM layer can later add timestamp-valid narrative information.

## Framing

The project forecasts predefined stress events, not undefined extreme events. A valid forecast is a probability attached to a timestamp, horizon, event definition, and immutable information set.

The practical use case is risk monitoring and research triage: convert heterogeneous market, options, credit, liquidity, macro, and document signals into a calibrated stress score with traceable evidence.

## Hypotheses

H1: Options, futures, credit, liquidity, and macro variables improve stress-regime forecasts relative to historical base rates and price-only benchmarks.

H2: Point-in-time financial narratives add incremental information beyond structured derivatives variables.

H3: The measured incremental value of LLM agents falls after controlling for parametric look-ahead bias, recognizable historical events, revised data, and document timestamp leakage.

## Panels

Panel 1 is the primary modern instrument-level study. It should use the longest consistent overlapping history for S&P spot or ETF proxies, futures, volatility indexes, option surfaces, rates, credit, liquidity, and macro series.

Panel 2 is a long-run historical regime extension. It can use reconstructed equity, bond, commodity, macro, banking-crisis, and political-event histories. It must not pretend modern listed options, VIX futures, or current credit indexes exist for 200-300 years.

## Benchmarks

The benchmark ladder is:

- unconditional historical frequency;
- rolling historical frequency;
- regularized logistic regression;
- rare-event or class-weighted logistic regression;
- discrete-time hazard model;
- regime-switching model;
- random forest;
- gradient-boosted trees;
- calibrated ensemble.

The primary benchmark should remain interpretable and hard to overfit. Logistic regression and hazard models are the first serious candidates.

## Agent Conditions

The agent experiments should include:

- structured-only benchmark;
- text-only point-in-time agent;
- deterministic structured-summary agent;
- hybrid point-in-time agent;
- sanitized-history agent;
- placebo-context agent;
- deliberately contaminated agent;
- numeric-only anonymized agent.

The contaminated agent is invalid for forecasting. Its job is to quantify how much performance can be inflated by unrestricted historical knowledge.

## Literature Anchors

Options-based tail-risk measurement is established. Cboe SKEW is explicitly tied to option-implied skewness of 30-day S&P 500 returns, and Kelly and Jiang connect tail-risk measures to asset prices and real activity.

LLM finance forecasting is active, but historical backtests face look-ahead risk. Glasserman and Lin show that GPT-based sentiment experiments can be contaminated by later return knowledge. More recent work including Look-Ahead-Bench, DatedGPT, FinCAD, and broader finance-LLM bias reviews makes the leakage-controlled comparison itself a meaningful contribution.

Useful starting points:

- https://www.cboe.com/us/indices/dashboard/skew/
- https://www.nber.org/papers/w19375
- https://arxiv.org/abs/2309.17322
- https://arxiv.org/abs/2601.13770
- https://arxiv.org/abs/2603.11838
- https://arxiv.org/abs/2605.24564
- https://arxiv.org/abs/2602.14233

## First-Pass Deliverable

The first pass should prove that the system can:

- generate synthetic point-in-time data;
- construct labels before model selection;
- expose a deliberate leakage trap;
- show random split inflation;
- run chronological and purged/embargoed validation;
- produce benchmark probability tables;
- enforce a strict agent schema;
- record enough metadata to reproduce the experiment.
