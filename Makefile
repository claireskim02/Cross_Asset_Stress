.PHONY: install test generate-synthetic demo audit clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

generate-synthetic:
	PYTHONPATH=src python -m cross_asset_stress.cli generate-synthetic

demo:
	PYTHONPATH=src python -m cross_asset_stress.cli run-experiment --config configs/base.yaml

audit:
	PYTHONPATH=src python -m cross_asset_stress.cli audit-leakage

clean:
	rm -rf .pytest_cache .ruff_cache

