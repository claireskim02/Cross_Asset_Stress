# Finaeon Setup

This note covers local authentication plumbing only. It does not define research-grade point-in-time field mappings yet.

The example scripts say modern endpoints are hosted at `https://api.finaeon.com`. Use an institutional proxy only if your network requires it and it works from a non-browser terminal session.

## Local Credentials

Fill in the ignored local `.env` file:

```bash
FINAEON_BASE_URL=https://api.finaeon.com
FINAEON_USERNAME=your_username
FINAEON_PASSWORD=your_password
FINAEON_BEARER_TOKEN=
FINAEON_TIMEOUT_SECONDS=30
```

Do not commit `.env`, API credentials, bearer tokens, or raw Finaeon exports.

For terminal calls, prefer:

```bash
FINAEON_BASE_URL=https://api.finaeon.com
```

An institutional proxy URL may require browser SSO and return HTML instead of an API token.

If terminal login is blocked but the web API console works, a short-lived bearer token can be placed in `FINAEON_BEARER_TOKEN` locally. Leave it blank otherwise.

## Test Login

```bash
PYTHONPATH=src python -m cross_asset_stress.cli finaeon login --env-file .env
```

By default the command reports token source and length, but does not print the token. To print it in a private terminal:

```bash
PYTHONPATH=src python -m cross_asset_stress.cli finaeon login --env-file .env --print-token
```

## Generic POST Call

Use the OpenAPI page to choose the endpoint and exact JSON payload. Example shape:

```bash
PYTHONPATH=src python -m cross_asset_stress.cli finaeon post \
  --endpoint /search \
  --payload-json '{"searchString":"AAPL","searchType":"symbol","baseFilter":"exactmatch","page":"1","pageSize":"25"}' \
  --output data/raw/finaeon_search_aapl.json \
  --env-file .env
```

The command automatically adds the bearer token to the JSON body after `/login`, matching Finaeon's modern endpoint examples.

Convenience wrapper:

```bash
PYTHONPATH=src python -m cross_asset_stress.cli finaeon search \
  --search-string AAPL \
  --search-type symbol \
  --base-filter exactmatch \
  --output data/raw/finaeon_search_aapl.json \
  --env-file .env
```

## Series Pull

Example:

```bash
PYTHONPATH=src python -m cross_asset_stress.cli finaeon series \
  --series-name AAPL \
  --start-date 01/01/2019 \
  --end-date 12/31/2025 \
  --periodicity Daily \
  --close-only \
  --output data/raw/finaeon_series_aapl_daily.json \
  --env-file .env
```

Trial access appears to cover full US stock history from 2019 onward and full history for the named trial tickers. Fundamentals are not available to trial users.

## Legacy GET Fallback

The older endpoints authenticate with username/password query parameters. Use only for debugging or export compatibility, and avoid logging URLs because credentials are embedded in the query string.

```bash
PYTHONPATH=src python -m cross_asset_stress.cli finaeon legacy-get \
  --endpoint /api/search.ashx \
  --params-json '{"searchstring":"AAPL","searchtype":"symbol","searchfilter":"exactmatch","type":"csv"}' \
  --output data/raw/finaeon_legacy_search_aapl.csv \
  --env-file .env
```

## Troubleshooting

If `finaeon login` reports `Token extraction source=html_response`, the configured base URL returned an HTML SSO page rather than a Finaeon token. This usually means the proxy needs an interactive browser session and cannot be used directly from a terminal.

If direct `https://api.finaeon.com` returns HTTP 403, the host is reachable but blocking this environment before the API login flow. In that case, use an authenticated network/proxy route or a short-lived bearer token obtained locally through the API console.

## Research Use Constraint

Before using Finaeon data in a valid backtest, document:

- endpoint;
- request body;
- returned fields;
- data timestamp semantics;
- time zone;
- adjustment policy;
- revision or vintage behavior;
- earliest valid prediction timestamp;
- local cache path and hash.
