# ChronoSwan

ChronoSwan is a research work in progress on point-in-time detection and attribution of market stress. The current draft tests whether cross-asset futures-options implied volatility, skew, and PCA concentration can identify stress regimes early enough to improve a simple S&P 500 exposure rule.

The project is framed as quantitative research, not as a claim to predict undefined extreme events. The empirical pipeline benchmarks VIX term structure, cross-asset option-implied stress, conditional event-rate lift, PCA absorption, and a 2020+ chronological holdout backtest. The intraday ES impulse workflow remains a case study for attributing large 60-minute moves after a stress regime has been detected.

Public raw data is not included. Bloomberg/Finaeon/Nasdaq workbooks, API credentials, parquet caches, executed notebooks, and generated reports are kept local and ignored by git.

## Current Research Page

- Public chartbook: https://claireskim02.github.io/ChronoSwan/docs/
- Source page: [docs/index.html](docs/index.html)
- Working paper notebook: [notebooks/Paper_Draft.ipynb](notebooks/Paper_Draft.ipynb)
- Curated paper exhibits: [paper_assets/](paper_assets/)
- Nasdaq Data Link setup: [docs/nasdaq_data_link.md](docs/nasdaq_data_link.md)
- Literature and benchmark map: [docs/literature_review.md](docs/literature_review.md)
- Research log: [docs/research_log.md](docs/research_log.md)
- Leakage protocol: [docs/leakage_protocol.md](docs/leakage_protocol.md)

## Local Execution

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
chronoswan pull-futures-options-ivm \
  --start-date 2007-01-01 \
  --raw-output data/raw/ar_ivm_extended_2007_current.parquet \
  --feature-output data/processed/ar_ivm_extended_constant_tenor_features_2007_current.parquet \
  --summary-output data/processed/ar_ivm_extended_coverage_2007_current.csv
jupyter nbconvert --to notebook --execute notebooks/Paper_Draft.ipynb --output ../reports/Paper_Draft_executed.ipynb
jupyter nbconvert --to html reports/Paper_Draft_executed.ipynb --output Paper_Draft_executed.html
```

The executed local research report is written to `reports/Paper_Draft_executed.html`.

## Research Boundary

Valid claims are limited to reproducible, point-in-time event definitions and benchmark comparisons. LLM agents are treated as future incremental information processors, not as unrestricted hindsight engines. Any agent result must record model identity, prompt/context hashes, data timestamps, and contamination controls before it can be compared with the structured market-data benchmarks.
