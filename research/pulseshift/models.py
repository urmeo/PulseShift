"""Climatology baseline and the logistic suppression model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import MODEL_FEATURES


def temporal_split(
    panel: pd.DataFrame,
    train_years: tuple[int, ...] = config.TRAIN_YEARS,
    test_year: int = config.TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[panel["year"].isin(train_years)].reset_index(drop=True)
    test = panel[panel["year"] == test_year].reset_index(drop=True)
    return train, test


class ClimatologyBaseline:
    """Suppression rate by season x daytype x hour; ignores weather."""

    keys = ["season", "daytype", "hour"]

    def fit(self, df: pd.DataFrame) -> ClimatologyBaseline:
        self.rates_ = df.groupby(self.keys)["suppressed"].mean()
        self.prior_ = df["suppressed"].mean()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        idx = list(zip(*[df[k] for k in self.keys]))
        p = np.array([self.rates_.get(k, self.prior_) for k in idx])
        return np.column_stack([1.0 - p, p])  # sklearn-style (n, 2)


def build_logistic(balanced: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced" if balanced else None
                ),
            ),
        ]
    )


def fit_logistic(
    train: pd.DataFrame, balanced: bool = True, features: list[str] = MODEL_FEATURES
) -> Pipeline:
    model = build_logistic(balanced)
    model.fit(train[features], train["suppressed"])
    return model


def fit_gbm(
    train: pd.DataFrame, features: list[str] = MODEL_FEATURES
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=0)
    model.fit(train[features], train["suppressed"])
    return model


def predict(
    model, df: pd.DataFrame, features: list[str] = MODEL_FEATURES
) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]
