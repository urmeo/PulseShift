"""Decision-curve net benefit."""

from __future__ import annotations

import numpy as np


def net_benefit(y_true, y_prob, thresholds) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    n = len(y_true)
    prevalence = y_true.mean()
    model, treat_all = [], []
    for pt in thresholds:
        flag = y_prob >= pt
        tp = np.sum(flag & (y_true == 1))
        fp = np.sum(flag & (y_true == 0))
        weight = pt / (1 - pt)
        model.append(tp / n - fp / n * weight)
        treat_all.append(prevalence - (1 - prevalence) * weight)
    return np.array(model), np.array(treat_all)
