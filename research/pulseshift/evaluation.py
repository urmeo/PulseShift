"""Discrimination and calibration metrics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_fit(y_true, y_prob):
    """Cox calibration: regress outcome on predicted logit."""
    model = LogisticRegression(max_iter=2000)
    model.fit(_logit(y_prob).reshape(-1, 1), y_true)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def expected_calibration_error(y_true, y_prob, n_bins=10):
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
    fn,
    n=1000,
    seed=0,
    alpha=0.05,
    require_two_classes=False,
    groups=None,
):
    """Percentile bootstrap CI; resamples clusters when groups is given."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if groups is not None:
        groups = np.asarray(groups)
        members = [np.where(groups == g)[0] for g in np.unique(groups)]

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
        vals.append(fn(y_true[s], y_prob[s]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 3), round(float(hi), 3)


def metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    slope, intercept = calibration_fit(y_true, y_prob)
    return {
        "n": int(len(y_true)),
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
