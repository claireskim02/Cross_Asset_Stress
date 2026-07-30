# ChronoSwan Research Log

ChronoSwan is a research work in progress on point-in-time detection and attribution of market stress. The repository is structured as a quantitative research log: data contracts, benchmark definitions, leakage controls, experiment notebooks, and implementation notes.

## Current Research Branches

- [Project README](../README.md)
- [Research design](research_design.md)
- [Literature context and benchmark map](literature_review.md)
- [Intraday ES impulse PCA note](intraday_impulse_pca.md)
- [Leakage protocol](leakage_protocol.md)
- [Event taxonomy](event_taxonomy.md)
- [Data dictionary](data_dictionary.md)
- [Experiment registry](experiment_registry.md)
- [Finaeon setup](finaeon_setup.md)

## Intraday Branch

The active intraday branch studies whether large 60-minute ES1 moves have recurring cross-asset drivers. The workflow:

- loads a local Bloomberg multi-sheet OHLCV workbook;
- defines large ES moves using a shifted rolling threshold;
- compares simple event-conditioned correlations with PCA factors;
- tracks rolling PCA concentration;
- runs a small chronological signal screen as a diagnostic benchmark.

Public pages should describe methodology and non-proprietary conclusions. Raw Bloomberg files, parquet caches, generated CSVs, and executed notebooks remain local unless licensing explicitly permits publication.
