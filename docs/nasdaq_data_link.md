# Nasdaq Data Link Data Audit

This note records the current Nasdaq Data Link path for the cross-asset correlation paper.

## Purpose

Nasdaq Data Link has two separate roles in this project:

- Generic time-series or EOD dataset discovery for prices and indices.
- The `AR/IVM` and `AR/IVS` Tables API product for futures-options implied volatility.

The second path is currently the useful one. It provides a long-history, cross-asset options-implied stress panel.

## Local Setup

Keep the API key in `.env`:

```bash
NASDAQ_DATA_LINK_API_KEY=...
```

Do not commit `.env` or paste the key into notebooks, docs, or chat logs.

## Probe Command

Run:

```bash
cross-asset-stress probe-nasdaq-data-link \
  --env-file .env \
  --output data/processed/nasdaq_data_link_probe.csv
```

The output is a sanitized entitlement/connectivity report. It records candidate dataset codes, HTTP status, row counts, and returned columns. It does not include the API key.

## Futures Options Pull

The attached product uses the Nasdaq Data Link Tables API:

- `AR/IVM`: compact implied-volatility model values.
- `AR/IVS`: full implied-volatility surfaces by delta.

Current first-stage pull uses `AR/IVM`:

```bash
cross-asset-stress pull-futures-options-ivm \
  --env-file .env \
  --start-date 2007-01-01 \
  --raw-output data/raw/ar_ivm_extended_2007_current.parquet \
  --feature-output data/processed/ar_ivm_extended_constant_tenor_features_2007_current.parquet \
  --summary-output data/processed/ar_ivm_extended_coverage_2007_current.csv
```

This writes local ignored files only. The derived feature panel keeps nearest 30d, 60d, and 90d values for selected-maturity futures price, ATM implied volatility, 25d/10d risk reversals, and 25d/10d butterflies.

Current pulled roots:

| Group | Roots |
| --- | --- |
| Equity index | ES, NQ, RTY |
| Rates | TU, FV, TY, TN, US, UB |
| FX | DX, AD, BP, CD, EC, JY, SF |
| Energy | CL, B, HO, NG, RB |
| Metals | GC, HG, SI |
| Agriculture | BO, C, S, SM, W |

The latest local expanded pull contains 2,979,966 raw `AR/IVM` rows and a 5,078 x 522 constant-tenor feature panel. Coverage starts in 2007 for most long-history roots; several roots begin later according to product availability.

## Candidate Dataset Codes

Initial candidate codes are defined in `src/cross_asset_stress/data/nasdaq_data_link.py`. They include possible CHRIS continuous-futures codes, EOD ETF codes, and a FRED VIX mirror. These are candidates, not guaranteed entitlements.

## Current Access Note

From this Codex environment, generic Data Link catalog search and direct CHRIS/EOD/FRED-style dataset probes returned HTTP 403, including harmless test datasets. The attached `AR/IVM` Tables API product is accessible and has been pulled successfully.

## Research Implication

Use `AR/IVM` for cross-asset option-implied stress, skew, and volatility-regime analysis. Use Bloomberg or another continuous-futures source when the paper needs clean underlying futures return correlations.
