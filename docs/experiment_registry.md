# Experiment Registry

This file is the human-readable registry stub. Machine-readable records are written to ignored files under `data/processed/`.

## Current Experiments

No committed empirical result should be treated as real-market evidence yet. The current synthetic demo validates plumbing, leakage checks, and benchmark execution.

## Required Fields

- experiment_id;
- git_commit;
- configuration hash;
- data snapshot or manifest hash;
- feature set;
- label definition;
- forecast horizon;
- train and test dates;
- purge and embargo settings;
- model and version;
- LLM provider and exact model identifier;
- declared model knowledge cutoff, if available;
- full prompts;
- context document IDs and hashes;
- random seed;
- metrics;
- calibration results;
- runtime and token usage;
- contamination flags.

