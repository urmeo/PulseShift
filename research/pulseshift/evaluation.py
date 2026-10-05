"""Discrimination and calibration metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def binary_predictions(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired binary labels and probabilities."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must be nonempty paired vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("probabilities must be finite and between 0 and 1")
    return y, p


def calibration_fit(y_true, y_prob) -> tuple[float, float]:
    """Cox calibration: unpenalized logistic of outcome on predicted logit."""
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if len(np.unique(y_true)) != 2:
        raise ValueError("calibration fitting requires both label classes")
    model = LogisticRegression(penalty=None, max_iter=2000)
    model.fit(_logit(y_prob).reshape(-1, 1), y_true)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def bootstrap_ci(
    y_true,
    y_prob,
    fn: Callable,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    require_two_classes: bool = False,
    groups=None,
) -> tuple[float, float]:
    """Percentile bootstrap CI; resamples clusters when groups is given."""
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 < alpha < 1:
        raise ValueError("bootstrap requires positive n and 0 < alpha < 1")
    if require_two_classes and len(np.unique(y_true)) < 2:
        raise ValueError("bootstrap metric requires both label classes")
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y_true):
            raise ValueError("bootstrap groups must align with labels")
        members = [np.where(groups == g)[0] for g in np.unique(groups)]
        if any(len(s) == 0 for s in members):
            raise ValueError("bootstrap groups must not contain missing values")

    def draw():
        if groups is None:
            return rng.choice(len(y_true), size=len(y_true), replace=True)
        picked = rng.integers(0, len(members), len(members))
        return np.concatenate([members[i] for i in picked])

    vals = []
    for _ in range(n):
        s = draw()
        if require_two_classes and len(np.unique(y_true[s])) < 2:
            continue
        value = float(fn(y_true[s], y_prob[s]))
        if not np.isfinite(value):
            raise ValueError("bootstrap metric returned a nonfinite value")
        vals.append(value)
    if not vals:
        raise ValueError("no valid bootstrap samples; increase n")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def metrics(y_true, y_prob) -> dict:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    slope, intercept = calibration_fit(y_true, y_prob)
    return {
        "n": len(y_true),
        "base_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(
            log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])
        ),
        "ece": expected_calibration_error(y_true, y_prob),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
