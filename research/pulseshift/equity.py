"""Subgroup stratification of burden and model performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config
from .evaluation import calibration_fit


def strata_metrics(df, group_col, risk_col="risk"):
    rows = []
    for name, g in df.groupby(group_col):
        y = g["suppressed"].to_numpy()
        p = g[risk_col].to_numpy()
        auroc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
        slope = calibration_fit(y, p)[0] if len(np.unique(y)) == 2 else np.nan
        rows.append(
            {
                "group": name,
                "n": len(g),
                "base_rate": float(y.mean()),
                "auroc": auroc,
                "brier": brier_score_loss(y, p) if len(np.unique(y)) == 2 else np.nan,
                "cal_slope": slope,
            }
        )
    return pd.DataFrame(rows)


def rider_burden(panel):
    """Suppression rate by rider type over the full 2022-2024 panel."""
    from .panel import _expected_rides

    rows = []
    for kind, col in [("member", "rides_member"), ("casual", "rides_casual")]:
        expected = _expected_rides(panel, value=col)
        active = expected >= config.EXPECTED_FLOOR
        suppressed = (
            panel[col].to_numpy() < config.SUPPRESSION_RATIO * expected
        ) & active
        rows.append(
            {
                "rider_type": kind,
                "active_hours": int(active.sum()),
                "suppression_rate": float(suppressed[active].mean()),
            }
        )
    return pd.DataFrame(rows)
