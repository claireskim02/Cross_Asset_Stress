"""Regularized logistic-regression benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cross_asset_stress.models.naive import HistoricalFrequencyModel


@dataclass
class LogisticBenchmarkResult:
    """Fitted benchmark and predicted probabilities."""

    model: Pipeline | HistoricalFrequencyModel
    probabilities: np.ndarray
    used_fallback: bool


def make_logistic_pipeline(
    *,
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Create the primary regularized logistic benchmark."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_predict_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
) -> LogisticBenchmarkResult:
    """Fit logistic regression and return event probabilities.

    If the training fold contains only one class, the function falls back to
    the historical-frequency benchmark. This keeps temporal folds explicit
    instead of silently dropping difficult rare-event windows.
    """

    target = pd.Series(y_train).dropna().astype(int)
    train = X_train.loc[target.index]
    if target.nunique() < 2:
        fallback = HistoricalFrequencyModel().fit(target)
        return LogisticBenchmarkResult(
            model=fallback,
            probabilities=fallback.predict_proba(len(X_test)),
            used_fallback=True,
        )

    model = make_logistic_pipeline(class_weight=class_weight, random_state=random_state)
    model.fit(train, target)
    probabilities = model.predict_proba(X_test)[:, 1]
    return LogisticBenchmarkResult(model=model, probabilities=probabilities, used_fallback=False)

