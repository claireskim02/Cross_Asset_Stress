# ChronoSwan

ChronoSwan is a research work in progress on point-in-time detection and attribution of market stress. The current empirical branch studies whether large 60-minute moves in ES1 can be explained by recurring cross-asset impulses across equity futures, rates, energy, metals, FX, volatility futures, and credit/rates ETFs.

The project is framed as quantitative research, not as a claim to predict unknowable black swans. The first objective is to define stress events without look-ahead bias, benchmark simple conditional correlations, test whether event-conditioned PCA adds useful factor structure, and document where a future LLM agent could help synthesize drivers after the structured benchmark is locked.

Public raw data is not included. Bloomberg/Finaeon workbooks, parquet caches, executed notebooks, and generated reports are kept local and ignored by git.

## Current Research Page

- Full public research page: [docs/index.html](docs/index.html)
- Intraday methodology note: [docs/intraday_impulse_pca.md](docs/intraday_impulse_pca.md)
- Main notebook: [notebooks/08_intraday_impulse_pca.ipynb](notebooks/08_intraday_impulse_pca.ipynb)
- Literature and benchmark map: [docs/literature_review.md](docs/literature_review.md)
- Leakage protocol: [docs/leakage_protocol.md](docs/leakage_protocol.md)

## Local Execution

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
chronoswan run-intraday-impulse-pca --input-path data/17sheets.xlsx
```

The executed local research report is written to `reports/intraday_impulse_pca_executed.html` after notebook execution.

## Research Boundary

Valid claims are limited to reproducible, point-in-time event definitions and benchmark comparisons. LLM agents are treated as future incremental information processors, not as unconstrained historical oracles. Any agent result must record model identity, prompt/context hashes, data timestamps, and contamination controls before it can be compared with the structured market-data benchmarks.
