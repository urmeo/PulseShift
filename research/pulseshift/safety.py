"""Audit that recommendations never raise exposure."""

from __future__ import annotations

import pandas as pd

from . import config


def audit(reco: pd.DataFrame, risk_col: str = "risk") -> dict:
    shifts = reco[reco["action"] == "shift"]
    risk_drop = shifts[risk_col] - shifts["chosen_risk"]

    unsafe_target = (reco["action"] != "cancel") & (
        (reco["target_heat_index_f"] >= config.HEAT_UNSAFE_F)
        | (reco["target_aqi"] >= config.AQI_UNSAFE)
    )
    return {
        "n_shifts": len(shifts),
        "n_cancel": int((reco["action"] == "cancel").sum()),
        "unsafe_recommendations": int(unsafe_target.sum()),
        "mean_risk_reduction": float(risk_drop.mean()) if len(shifts) else 0.0,
        "all_safe": bool(unsafe_target.sum() == 0),
    }
