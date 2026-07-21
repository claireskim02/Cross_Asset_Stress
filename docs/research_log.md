# Research Log

Status: scaffold initialized. No empirical market result yet. Current outputs are synthetic checks of pipeline mechanics, validation rules, and leakage diagnostics.

## Research Position

The viable project is not "predict black swans." The viable project is to test whether point-in-time text and agent reasoning add incremental information to derivatives-implied crash-risk measures after leakage controls.

Working question:

Can a leakage-controlled, point-in-time LLM agent improve calibration or event-level detection of predefined U.S. equity-market stress regimes beyond structured derivatives, futures, credit, liquidity, and macro benchmarks?

## Current Decisions

Use synthetic data first. This prevents vendor-schema guessing and gives known leakage traps for tests.

Use a strict point-in-time data contract before loading Bloomberg or Finaeon exports.

Use regularized logistic regression as the first serious benchmark. It is interpretable, relatively hard to overfit, and appropriate for calibration diagnostics.

Use a mock agent as the default agent. It is free, local, deterministic, and schema-valid. Add Ollama as an optional local open-weight path, not as a test dependency.

Treat contaminated LLM runs as diagnostic upper bounds only.

## Data Notes

No external data is required for the current scaffold.

yfinance can be useful for a public-data pass-through test, especially SPY, index proxies, and VIX-like series. It should not be treated as research-grade point-in-time data because historical adjustments, revision metadata, and release timing are not controlled enough for the final claim.

Bloomberg and Finaeon are likely useful for the serious pass, but only after defining:

- exact ticker and field lists;
- adjustment policy;
- time zone and market-close convention;
- publication or release timestamp;
- ingestion timestamp;
- vintage or snapshot identifier;
- licensing boundary for cached files.

No API keys should be pasted into chat or committed. Use `.env` locally.

Finaeon auth plumbing is now available through `chronoswan finaeon login`, with explicit wrappers for `/search` and `/series`. The example scripts indicate that modern authenticated calls pass the bearer token inside the JSON request body. PIT field mappings remain research tasks, not assumptions to be filled in by code.

## Agent Notes

The current default is `MockAgentForecaster`. It makes no API calls and costs nothing.

The optional Ollama route is for local open-weight experiments once an Ollama server is running. This does not solve parametric look-ahead bias by itself. It only avoids paid API dependency and makes model selection reproducible on local infrastructure.

The valid agent record must store:

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
- rolling event frequency;
- regularized logistic regression;
- class-weighted logistic regression;
- discrete-time hazard model;
- regime-switching model;
- tree ensemble;
- calibrated ensemble.

The first pass currently implements unconditional frequency and regularized logistic regression.

## Open Questions

What is the primary event family: forward drawdown, volatility regime, VIX threshold, joint stress, or time to event?

What horizon is primary: 20 trading days is a practical initial default, but 5, 60, and 120 days answer different risk-management questions.

Should Bloomberg/Finaeon be used first for structured data, or should the next step be a public pass-through dataset to debug field handling?

Which local open-weight model should be the first Ollama baseline?

How should document archives be sourced so first-seen timestamps are defensible?

## Near-Term Next Step

Run the synthetic notebook, inspect the leakage table, and decide the first real-data slice. A practical first slice is daily SPY or SPX proxy, VIX, VIX term structure if available, basic Treasury/credit proxies, and a small point-in-time document set.
