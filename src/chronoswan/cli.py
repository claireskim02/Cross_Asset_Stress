"""ChronoSwan command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from chronoswan.agents.forecaster import MockAgentForecaster, OllamaAgentForecaster
from chronoswan.data.finaeon import FinaeonAPIError, FinaeonClient, FinaeonConfigError
from chronoswan.data.point_in_time import build_point_in_time_feature_matrix
from chronoswan.data.synthetic import (
    CLEAN_FEATURES,
    generate_synthetic_feature_store,
    generate_synthetic_market_frame,
    write_synthetic_dataset,
)
from chronoswan.events.labels import build_event_labels
from chronoswan.experiments.bloomberg_quicklook import run_bloomberg_quicklook
from chronoswan.experiments.literature_benchmarks import run_literature_benchmark_replication
from chronoswan.experiments.registry import load_yaml
from chronoswan.experiments.runner import run_synthetic_demo
from chronoswan.validation.leakage_checks import audit_feature_matrix

app = typer.Typer(help="ChronoSwan research CLI.", no_args_is_help=True)
finaeon_app = typer.Typer(help="Finaeon API utilities.")
app.add_typer(finaeon_app, name="finaeon")


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Annotated[Path, typer.Option(help="Directory for generated parquet files.")] = Path(
        "data/synthetic"
    ),
    start: Annotated[str, typer.Option(help="Start date.")] = "1995-01-02",
    end: Annotated[str, typer.Option(help="End date.")] = "2025-12-31",
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = 42,
) -> None:
    """Generate deterministic synthetic point-in-time data."""

    paths = write_synthetic_dataset(output_dir, start=start, end=end, seed=seed)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command("validate-data")
def validate_data(
    features_path: Annotated[Path, typer.Option(help="Long PIT feature parquet.")] = Path(
        "data/synthetic/synthetic_pit_features.parquet"
    ),
) -> None:
    """Validate synthetic feature store shape and known leakage diagnostics."""

    if not features_path.exists():
        typer.echo("Feature file not found; generating synthetic data first.")
        write_synthetic_dataset(features_path.parent)
    features = pd.read_parquet(features_path)
    required = {
        "event_time",
        "observation_time",
        "release_time",
        "ingestion_time",
        "earliest_valid_prediction_timestamp",
        "feature_name",
        "value",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise typer.BadParameter(f"Missing required feature columns: {missing}")
    typer.echo(f"Validated {len(features):,} long-form feature rows.")


@app.command("build-features")
def build_features(
    as_of: Annotated[str, typer.Option(help="Forecast timestamp, for example 2020-03-02.")],
) -> None:
    """Build a one-row point-in-time feature matrix for an as-of date."""

    features = generate_synthetic_feature_store()
    matrix = build_point_in_time_feature_matrix(
        [pd.Timestamp(as_of, tz="UTC")],
        features,
        feature_names=CLEAN_FEATURES,
    )
    typer.echo(matrix.to_string(index=False))


@app.command("build-labels")
def build_labels(
    output_path: Annotated[Path, typer.Option(help="Output parquet path.")] = Path(
        "data/synthetic/synthetic_labels.parquet"
    ),
) -> None:
    """Build synthetic labels."""

    market = generate_synthetic_market_frame()
    labels = build_event_labels(market)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)
    typer.echo(f"Wrote labels: {output_path}")


@app.command("run-baselines")
def run_baselines() -> None:
    """Run the synthetic benchmark suite."""

    result = run_synthetic_demo()
    typer.echo(result["metrics"].to_string(index=False))


@app.command("run-agent")
def run_agent(
    provider: Annotated[str, typer.Option(help="Agent provider: mock or ollama.")] = "mock",
    model: Annotated[str | None, typer.Option(help="Local model ID for Ollama.")] = None,
) -> None:
    """Run a point-in-time agent on a synthetic observation."""

    market = generate_synthetic_market_frame()
    row = market.iloc[-60]
    if provider == "mock":
        agent = MockAgentForecaster()
    elif provider == "ollama":
        agent = OllamaAgentForecaster(model_id=model)
    else:
        raise typer.BadParameter("provider must be 'mock' or 'ollama'")
    forecast = agent.forecast(
        as_of_timestamp=row["forecast_timestamp"].to_pydatetime(),
        structured_features={feature: float(row[feature]) for feature in CLEAN_FEATURES},
    )
    typer.echo(forecast.model_dump_json(indent=2))


@app.command("audit-leakage")
def audit_leakage() -> None:
    """Run leakage checks against the synthetic demo matrix."""

    result = run_synthetic_demo()
    findings = result["leakage_findings"]
    if not findings:
        typer.echo("No leakage findings.")
        return
    for finding in findings:
        typer.echo(f"[{finding.severity}] {finding.check} {finding.column}: {finding.message}")


@app.command("evaluate")
def evaluate() -> None:
    """Alias for the synthetic evaluation table."""

    run_baselines()


@app.command("run-experiment")
def run_experiment(
    config: Annotated[Path, typer.Option(help="YAML experiment config.")] = Path("configs/base.yaml"),
) -> None:
    """Run a configured synthetic experiment."""

    cfg = load_yaml(config)
    result = run_synthetic_demo(cfg)
    typer.echo(result["metrics"].to_string(index=False))
    typer.echo(f"Outputs: {result.get('output_dir', 'not written')}")


@app.command("run-bloomberg-quicklook")
def run_bloomberg_quicklook_command(
    input_path: Annotated[Path, typer.Option(help="Manual Bloomberg workbook path.")] = Path(
        "data/raw/Book1.xlsx"
    ),
    output_dir: Annotated[Path, typer.Option(help="Output directory for derived tables.")] = Path(
        "data/processed"
    ),
) -> None:
    """Run the first Bloomberg case-study and benchmark pass."""

    result = run_bloomberg_quicklook(input_path=input_path, output_dir=output_dir)
    panel = result["panel"]
    typer.echo(
        f"Bloomberg quicklook panel: {len(panel):,} rows, "
        f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    )
    typer.echo("\nEvent base rates:")
    typer.echo(result["base_rates"].to_string(index=False))
    typer.echo("\nCase studies:")
    typer.echo(result["case_studies"].round(3).to_string(index=False))
    typer.echo("\nHoldout model results:")
    typer.echo(result["model_results"].round(4).to_string(index=False))
    typer.echo(f"\nOutputs: {result['output_dir']}")


@app.command("run-literature-benchmarks")
def run_literature_benchmarks_command(
    input_path: Annotated[Path, typer.Option(help="Manual Bloomberg workbook path.")] = Path(
        "data/raw/Book1.xlsx"
    ),
    output_dir: Annotated[Path, typer.Option(help="Output directory for derived tables.")] = Path(
        "data/processed"
    ),
) -> None:
    """Run pre-LLM literature-inspired structured benchmarks."""

    result = run_literature_benchmark_replication(input_path=input_path, output_dir=output_dir)
    panel = result["panel"]
    typer.echo(
        f"Literature benchmark panel: {len(panel):,} rows, "
        f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    )
    typer.echo("\nConditional indicator lifts:")
    typer.echo(result["conditional_rates"].round(4).to_string(index=False))
    typer.echo("\nFeature-group holdout:")
    typer.echo(result["model_results"].round(4).to_string(index=False))
    typer.echo(f"\nOutputs: {result['output_dir']}")


@finaeon_app.command("login")
def finaeon_login(
    env_file: Annotated[Path, typer.Option(help="Local env file with Finaeon credentials.")] = Path(
        ".env"
    ),
    print_token: Annotated[bool, typer.Option(help="Print bearer token. Avoid in shared logs.")] = False,
) -> None:
    """Authenticate with Finaeon using local environment credentials."""

    try:
        client = FinaeonClient.from_env(env_file=env_file)
        result = client.login()
    except (FinaeonConfigError, FinaeonAPIError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Finaeon login OK: HTTP {result.status_code}, token source={result.token_source}, "
        f"token length={len(result.token)}"
    )
    if print_token:
        typer.echo(result.token)


@finaeon_app.command("post")
def finaeon_post(
    endpoint: Annotated[str, typer.Option(help="Endpoint path, for example /search or /series.")],
    payload_json: Annotated[
        str | None,
        typer.Option(help="Inline JSON payload. Use this or --payload-file."),
    ] = None,
    payload_file: Annotated[
        Path | None,
        typer.Option(help="Path to JSON payload file. Use this or --payload-json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional output path for the JSON/text response."),
    ] = None,
    env_file: Annotated[Path, typer.Option(help="Local env file with Finaeon credentials.")] = Path(
        ".env"
    ),
) -> None:
    """POST a user-supplied JSON payload to an authenticated Finaeon endpoint."""

    if (payload_json is None) == (payload_file is None):
        raise typer.BadParameter("Provide exactly one of --payload-json or --payload-file")

    payload_text = payload_file.read_text(encoding="utf-8") if payload_file else payload_json
    assert payload_text is not None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Finaeon POST payload must be a JSON object")

    try:
        client = FinaeonClient.from_env(env_file=env_file)
        response = client.post(endpoint, payload)
    except (FinaeonConfigError, FinaeonAPIError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    rendered = response if isinstance(response, str) else json.dumps(response, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered or "", encoding="utf-8")
        typer.echo(f"Wrote Finaeon response: {output}")
    else:
        typer.echo(rendered)


@finaeon_app.command("search")
def finaeon_search(
    search_string: Annotated[str, typer.Option(help="Symbol/name search string, e.g. AAPL.")],
    search_type: Annotated[str, typer.Option(help="Search key: symbol or name.")] = "symbol",
    base_filter: Annotated[
        str,
        typer.Option(help="Match mode: exactmatch, contains, or startswith."),
    ] = "exactmatch",
    page: Annotated[int, typer.Option(help="Result page.")] = 1,
    page_size: Annotated[int, typer.Option(help="Page size, typically 10-100.")] = 25,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional output path for JSON/text response."),
    ] = None,
    env_file: Annotated[Path, typer.Option(help="Local env file with Finaeon credentials.")] = Path(
        ".env"
    ),
) -> None:
    """Search Finaeon series metadata using POST /search."""

    try:
        response = FinaeonClient.from_env(env_file=env_file).search(
            search_string,
            search_type=search_type,
            base_filter=base_filter,
            page=page,
            page_size=page_size,
        )
    except (FinaeonConfigError, FinaeonAPIError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit_finaeon_response(response, output=output)


@finaeon_app.command("series")
def finaeon_series(
    series_name: Annotated[str, typer.Option(help="Series symbol/ticker, comma-separated allowed.")],
    start_date: Annotated[
        str | None,
        typer.Option(help="Start date in MM/DD/YYYY format."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="End date in MM/DD/YYYY format."),
    ] = None,
    periodicity: Annotated[str, typer.Option(help="Daily, Weekly, Monthly, Quarterly, Annual.")] = "Daily",
    split_adjusted: Annotated[bool, typer.Option(help="Return split-adjusted prices.")] = True,
    close_only: Annotated[bool, typer.Option(help="Return close-only data.")] = False,
    metadata: Annotated[bool, typer.Option(help="Include series metadata.")] = True,
    point_in_time: Annotated[
        bool,
        typer.Option(help="Use Finaeon's pointInTime flag for split application."),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional output path for JSON/text response."),
    ] = None,
    env_file: Annotated[Path, typer.Option(help="Local env file with Finaeon credentials.")] = Path(
        ".env"
    ),
) -> None:
    """Request Finaeon price data using POST /series."""

    try:
        response = FinaeonClient.from_env(env_file=env_file).series(
            series_name,
            start_date=start_date,
            end_date=end_date,
            periodicity=periodicity,
            split_adjusted=split_adjusted,
            close_only=close_only,
            metadata=metadata,
            point_in_time=point_in_time,
        )
    except (FinaeonConfigError, FinaeonAPIError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit_finaeon_response(response, output=output)


@finaeon_app.command("legacy-get")
def finaeon_legacy_get(
    endpoint: Annotated[
        str,
        typer.Option(help="Legacy endpoint, e.g. /api/search.ashx or /api/api.ashx."),
    ],
    params_json: Annotated[
        str,
        typer.Option(help="JSON object of legacy query params, excluding username/password."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Optional output path for text/csv response."),
    ] = None,
    env_file: Annotated[Path, typer.Option(help="Local env file with Finaeon credentials.")] = Path(
        ".env"
    ),
) -> None:
    """Call a Finaeon legacy GET endpoint with username/password query auth."""

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Params are not valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise typer.BadParameter("--params-json must be a JSON object")
    try:
        response = FinaeonClient.from_env(env_file=env_file).legacy_get(endpoint, params)
    except (FinaeonConfigError, FinaeonAPIError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit_finaeon_response(response, output=output)


def _emit_finaeon_response(
    response: dict[str, object] | list[object] | str | None,
    *,
    output: Path | None,
) -> None:
    rendered = response if isinstance(response, str) else json.dumps(response, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered or "", encoding="utf-8")
        typer.echo(f"Wrote Finaeon response: {output}")
    else:
        typer.echo(rendered)


if __name__ == "__main__":
    app()
