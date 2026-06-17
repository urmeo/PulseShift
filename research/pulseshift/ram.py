"""Time-shift adaptation and Recovered Active Minutes (RAM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _safe(heat_index_f, aqi):
    return (heat_index_f < config.HEAT_UNSAFE_F) & (aqi < config.AQI_UNSAFE)


def recommend(
    panel: pd.DataFrame, risk_col: str = "risk", window: int = config.SHIFT_WINDOW_H
) -> pd.DataFrame:
    """Per active hour, choose keep or the lowest-risk safe time shift."""
    df = panel.copy()
    df["day"] = df["ts_local"].dt.date
    df["safe"] = _safe(df["heat_index_f"], df["aqi"])

    actions, targets, target_risk, t_heat, t_aqi = [], [], [], [], []
    for _, day in df.groupby("day"):
        hours = day["hour"].to_numpy()
        risk = day[risk_col].to_numpy()
        safe = day["safe"].to_numpy()
        heat = day["heat_index_f"].to_numpy()
        air = day["aqi"].to_numpy()
        for i in range(len(day)):
            near = np.abs(hours - hours[i]) <= window
            feasible = near & safe  # stay within the safe envelope
            if not feasible.any():
                actions.append("cancel")
                targets.append(np.nan)
                target_risk.append(1.0)
                t_heat.append(heat[i])
                t_aqi.append(air[i])
                continue
            cand = np.where(feasible)[0]
            best = cand[np.argmin(risk[cand])]
            if not safe[i]:
                action = "shift"  # must leave an unsafe hour
            elif best == i or (risk[i] - risk[best]) < config.MIN_RISK_BENEFIT:
                action, best = "keep", i  # benefit too small to bother
            else:
                action = "shift"
            actions.append(action)
            targets.append(hours[best])
            target_risk.append(risk[best])
            t_heat.append(heat[best])
            t_aqi.append(air[best])

    df["action"] = actions
    df["target_hour"] = targets
    df["chosen_risk"] = target_risk
    df["target_heat_index_f"] = t_heat
    df["target_aqi"] = t_aqi
    return df


def ram_table(reco: pd.DataFrame, risk_col: str = "risk") -> dict:
    """Recovered activity from acting vs doing nothing."""
    expected = reco["expected_rides"].to_numpy()
    no_adapt = expected * (1 - reco[risk_col].to_numpy())
    adapted = expected * (1 - reco["chosen_risk"].to_numpy())
    recovered = np.maximum(0.0, adapted - no_adapt)

    lost = float((expected * reco[risk_col].to_numpy()).sum())
    total = float(recovered.sum())
    shifts = reco[reco["action"] == "shift"]
    distinct_slots = (
        int(shifts.groupby(["day", "target_hour"]).ngroups) if len(shifts) else 0
    )
    return {
        "recovered_rides": total,
        "recovered_minutes": total * config.MEAN_RIDE_MIN,
        "lost_rides_no_adapt": lost,
        "ram_pct_of_lost": float(total / lost) if lost else 0.0,
        "share_shifted": float((reco["action"] == "shift").mean()),
        "share_cancel": float((reco["action"] == "cancel").mean()),
        "n_shifts": len(shifts),
        "distinct_target_slots": distinct_slots,
        "per_hour": pd.Series(recovered, index=reco.index),
    }
