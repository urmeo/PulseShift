"""Audit the configured heat and AQI limits for recommendation targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def audit(reco: pd.DataFrame, risk_col: str = "risk") -> dict:
    shifts = reco[reco["action"] == "shift"]
    risk_drop = shifts[risk_col] - shifts["chosen_risk"]

    applicable = reco["action"] != "cancel"
    invalid_target = applicable & (
        ~np.isfinite(reco["target_heat_index_f"])
        | ~np.isfinite(reco["target_aqi"])
        | ~np.isfinite(reco["chosen_risk"])
        | ~reco["chosen_risk"].between(0, 1)
    )
    unsafe_target = applicable & (
        (reco["target_heat_index_f"] >= config.HEAT_UNSAFE_F)
        | (reco["target_aqi"] >= config.AQI_UNSAFE)
        | invalid_target
    )
    return {
        "n_shifts": len(shifts),
        "n_cancel": int((reco["action"] == "cancel").sum()),
        "unsafe_recommendations": int(unsafe_target.sum()),
        "invalid_recommendations": int(invalid_target.sum()),
        "mean_risk_reduction": float(risk_drop.mean()) if len(shifts) else 0.0,
        "all_safe": bool(unsafe_target.sum() == 0),
    }
