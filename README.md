# Cross-Asset Stress Monitor

Cross-Asset Stress Monitor is a research work in progress on point-in-time detection and attribution of market stress. The current draft tests whether cross-asset futures-options implied volatility, skew, and PCA concentration can identify stress regimes early enough to improve a simple S&P 500 exposure rule.

The project is framed as quantitative research on predefined stress regimes rather than open-ended extreme-event prediction. The empirical pipeline benchmarks VIX term structure, cross-asset option-implied stress, conditional event-rate lift, PCA absorption, and a 2020+ chronological holdout backtest. The intraday ES impulse workflow remains a case study for attributing large 60-minute moves after a stress regime has been detected.

Public raw data is not included. Bloomberg/Finaeon/Nasdaq workbooks, API credentials, parquet caches, executed notebooks, and generated reports are kept local and ignored by git.

## Current Research Page

- Public chartbook: docs/
- Source page: [docs/index.html](docs/index.html)
- Working paper notebook: [notebooks/cross_asset_stress_monitor_paper.ipynb](notebooks/cross_asset_stress_monitor_paper.ipynb)
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
cross-asset-stress pull-futures-options-ivm \
  --start-date 2007-01-01 \
  --raw-output data/raw/ar_ivm_extended_2007_current.parquet \
  --feature-output data/processed/ar_ivm_extended_constant_tenor_features_2007_current.parquet \
  --summary-output data/processed/ar_ivm_extended_coverage_2007_current.csv
jupyter nbconvert --to notebook --execute notebooks/cross_asset_stress_monitor_paper.ipynb --output ../reports/cross_asset_stress_monitor_paper_executed.ipynb
jupyter nbconvert --to html reports/cross_asset_stress_monitor_paper_executed.ipynb --output cross_asset_stress_monitor_paper_executed.html
```

The executed local research report is written to `reports/cross_asset_stress_monitor_paper_executed.html`.

## Research Boundary

The current evidence is limited to reproducible, point-in-time event definitions and benchmark comparisons. LLM agents are treated as future incremental information processors, not as unrestricted hindsight engines. Any agent result must record model identity, prompt/context hashes, data timestamps, and contamination controls before it can be compared with the structured market-data benchmarks.
