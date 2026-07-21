# ChronoSwan

ChronoSwan is a research work in progress on point-in-time detection of extreme U.S. equity-market stress using derivatives-market signals, historical regimes, and leakage-aware LLM agents.

The project does not claim to predict unknowable black swans. The defensible target is a calibrated probability that the market enters a predefined downside, volatility, or systemic-stress regime over a future horizon.

This repository should read as a quantitative research log: assumptions, data contracts, benchmark lineage, leakage controls, and reproducible experiment notes come before model complexity.

Working notes:

- [docs/research_log.md](docs/research_log.md)
- [docs/literature_review.md](docs/literature_review.md)
- [docs/research_design.md](docs/research_design.md)
- [docs/leakage_protocol.md](docs/leakage_protocol.md)
- [docs/finaeon_setup.md](docs/finaeon_setup.md)
- [notebooks/00_research_runthrough.ipynb](notebooks/00_research_runthrough.ipynb)
- [notebooks/07_literature_benchmark_replication.ipynb](notebooks/07_literature_benchmark_replication.ipynb)

## Research Questions

1. Do options, futures, credit, liquidity, and macro variables forecast defined stress regimes better than price-only or base-rate benchmarks?
2. Does a point-in-time LLM agent add incremental information beyond structured derivatives-market signals?
3. How much apparent LLM forecasting skill disappears after controlling for document leakage, revised data, parametric memory, recognizable event names, and overfit backtests?

## Current Scaffold

This first pass is intentionally synthetic and local. It includes:

- point-in-time Pydantic schemas for feature and document records;
- deterministic synthetic market data with publication timestamps;
- deliberate leaked features for audit tests;
- forward drawdown, volatility, VIX-like, joint-stress, event-phase, and time-to-event labels;
- chronological and purged/embargoed time-series splits;
- historical-frequency and regularized logistic-regression benchmarks;
- a strict agent-output schema, a free local mock agent, and an optional Ollama interface;
- CLI commands for synthetic generation, baseline runs, leakage audits, and evaluation;
- a main walkthrough notebook: [notebooks/00_research_runthrough.ipynb](notebooks/00_research_runthrough.ipynb).

No paid data APIs or paid model APIs are connected. No proprietary data should be committed.

## Valid And Invalid Claims

Valid:

- "This model estimates the probability of a predefined market-stress event."
- "This experiment compares point-in-time text and structured market indicators under explicit leakage controls."
- "The contaminated condition measures how much performance can be inflated by invalid future information."

Invalid:

- "This predicts black swans."
- "An unrestricted historical LLM result is evidence of predictive ability."
- "Prompt instructions alone remove LLM look-ahead bias."
- "Latest revised macro data can be used in historical forecasts."

Every valid forecast must be reproducible from an immutable information set available at the forecast timestamp.

## Study Architecture

The core comparison is:

- market-only statistical benchmark;
- structured market-data machine-learning model;
- point-in-time text-only agent;
- deterministic structured-summary agent;
- hybrid point-in-time agent;
- sanitized-history agent;
- placebo-context agent;
- deliberately contaminated agent used only as an invalid diagnostic upper bound;
- numeric-only anonymized experiment.

The first executable scaffold covers the structured benchmark, contamination diagnostics, and mock-agent plumbing.

## Event Definitions

Candidate targets are configured rather than selected after seeing results:

- forward S&P 500 drawdowns exceeding 5, 10, 15, or 20 percent;
- forward realized volatility entering top 5 percent or top 1 percent regimes;
- VIX or VIX-like thresholds within the forecast horizon;
- joint stress events combining drawdown, volatility, liquidity, and credit conditions;
- time to next event onset for survival or hazard modeling.

Supported horizons are 1, 5, 20, 60, and 120 trading days. Labels distinguish onset, continuation, recovery, and overlapping event windows.

## Leakage Taxonomy

ChronoSwan treats leakage as a first-class experimental variable:

- future target leakage from engineered features;
- timestamp leakage from document retrieval;
- revised macro data used in place of historical vintages;
- survivorship bias in assets or documents;
- famous event names that reveal historical identity;
- parametric memory in LLMs trained after the backtest period;
- random splits that let overlapping forward windows contaminate validation.

See [docs/leakage_protocol.md](docs/leakage_protocol.md).

## Benchmark Models

Implemented now:

- historical unconditional event frequency;
- regularized logistic regression with class weighting;
- random-split leakage diagnostic;
- chronological holdout;
- purged and embargoed time-series validation.

Planned:

- rolling historical frequency in experiment tables;
- rare-event logistic variants;
- discrete-time hazard models;
- HMM/regime-switching models;
- random forests and gradient-boosted trees;
- calibrated ensembles.

## Agent Experiments

The current agent is `MockAgentForecaster`, a deterministic, free, local agent that returns JSON matching the strict schema. It makes no API calls.

Future open-source options can be wired through Ollama, vLLM, or another local inference server using models such as Llama, Qwen, Mistral, or DeepSeek-family open-weight models. The scaffold includes an optional Ollama interface, but the default remains the mock agent so tests do not require a model server. These integrations should preserve prompt hashes, context hashes, model identifiers, and declared knowledge cutoffs where available.

Paid models can be evaluated later, but only as additional experimental conditions.

## Data Plan

Stage 0 uses synthetic data only.

Stage 1 can add public-data pass-through adapters, for example yfinance for SPY or index proxies and public VIX-like series. Treat this as pipeline-debugging data, not research-grade point-in-time evidence.

Stage 2 should use vendor-quality sources:

- Bloomberg or Finaeon for market, futures, macro, and cross-asset histories;
- OptionMetrics, Cboe, or licensed vendor exports for option surfaces, skew, and tail measures;
- ALFRED/FRED-style vintages for public macro revisions when applicable;
- local document archives with original publication and first-seen timestamps.

Put API keys in `.env`, never in chat, code, notebooks, or git history. Raw data and generated caches are ignored by git.

## Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Generate synthetic data:

```bash
chronoswan generate-synthetic
```

Run the first synthetic demonstration:

```bash
chronoswan run-experiment --config configs/base.yaml
```

Without installing the package:

```bash
PYTHONPATH=src python -m chronoswan.cli run-experiment --config configs/base.yaml
```

## CLI Commands

```bash
chronoswan generate-synthetic
chronoswan validate-data
chronoswan build-features --as-of 2020-03-02
chronoswan build-labels
chronoswan run-baselines
chronoswan run-agent
chronoswan audit-leakage
chronoswan evaluate
chronoswan run-experiment --config configs/base.yaml
chronoswan run-bloomberg-quicklook --input-path data/raw/Book1.xlsx
chronoswan run-literature-benchmarks --input-path data/raw/Book1.xlsx
chronoswan finaeon login --env-file .env
chronoswan finaeon search --search-string AAPL --search-type symbol --base-filter exactmatch
chronoswan finaeon series --series-name AAPL --start-date 01/01/2019 --periodicity Daily
```

## Seed Literature

This is not an exhaustive literature review, but it anchors the first benchmark set:

- Cboe SKEW Index estimates 30-day S&P 500 return-distribution skewness from option prices: https://www.cboe.com/us/indices/dashboard/skew/
- Kelly and Jiang, "Tail Risk and Asset Prices", links tail-risk measures to option-implied skewness/kurtosis and asset prices: https://www.nber.org/papers/w19375
- Glasserman and Lin study look-ahead bias in GPT-based financial sentiment backtests: https://arxiv.org/abs/2309.17322
- Look-Ahead-Bench proposes standardized point-in-time LLM bias evaluation for finance: https://arxiv.org/abs/2601.13770
- DatedGPT trains cutoff-specific models to reduce look-ahead bias: https://arxiv.org/abs/2603.11838
- FinCAD proposes inference-time mitigation for parametric look-ahead bias: https://arxiv.org/abs/2605.24564
- A 2026 finance-LLM bias position paper highlights look-ahead, survivorship, narrative, objective, and cost biases: https://arxiv.org/abs/2602.14233

## Known Limitations

- Synthetic results prove only that the pipeline runs and catches known leakage traps.
- yfinance-style data is convenient but not enough for formal point-in-time claims.
- Modern options and volatility data do not support a homogeneous 200-300 year panel.
- Extreme events are rare and overlapping, so event-level evaluation is required.
- LLM historical backtests can be invalid even when supplied documents are timestamp-valid, because the model may already know later outcomes.

## Planned Research Stages

1. Finish synthetic demonstration and main walkthrough notebook.
2. Add public pass-through adapters and compare against synthetic behavior.
3. Add Bloomberg/Finaeon local adapters after field metadata and release timing are documented.
4. Add option-implied tail-risk features and crisis-level evaluation.
5. Wire a local open-weight LLM condition behind the strict agent schema.
6. Add sanitized, placebo, contaminated, and anonymized agent experiments.
7. Produce a reproducible research memo with benchmark tables, calibration plots, and next-stage hypotheses.
