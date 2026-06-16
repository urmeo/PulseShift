"""Climatology baseline and the logistic suppression model."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import MODEL_FEATURES


def temporal_split(panel, train_years=config.TRAIN_YEARS, test_year=config.TEST_YEAR):
    train = panel[panel["year"].isin(train_years)].reset_index(drop=True)
    test = panel[panel["year"] == test_year].reset_index(drop=True)
    return train, test


class ClimatologyBaseline:
    """Suppression rate by season x daytype x hour; ignores weather."""

    keys = ["season", "daytype", "hour"]

    def fit(self, df):
        self.rates_ = df.groupby(self.keys)["suppressed"].mean()
        self.prior_ = df["suppressed"].mean()
        return self

    def predict_proba(self, df):
        idx = list(zip(*[df[k] for k in self.keys]))
        return np.array([self.rates_.get(k, self.prior_) for k in idx])


def build_logistic(balanced=True):
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


def fit_logistic(train, balanced=True, features=MODEL_FEATURES):
    model = build_logistic(balanced)
    model.fit(train[features], train["suppressed"])
    return model


def fit_gbm(train, features=MODEL_FEATURES):
    model = GradientBoostingClassifier(random_state=0)
    model.fit(train[features], train["suppressed"])
    return model


def predict(model, df, features=MODEL_FEATURES):
    return model.predict_proba(df[features])[:, 1]
