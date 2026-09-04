"""Ablation definitions for later model comparisons."""

from __future__ import annotations


DEFAULT_ABLATIONS = {
    "structured_clean": {
        "features": "clean structured features only",
        "valid_forecast": True,
    },
    "structured_leaky_diagnostic": {
        "features": "clean features plus deliberate synthetic leakage traps",
        "valid_forecast": False,
    },
}
