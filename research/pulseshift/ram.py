"""Time-shift adaptation and Recovered Active Minutes (RAM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _safe(heat_index_f, aqi):
    return (
        np.isfinite(heat_index_f)
        & np.isfinite(aqi)
        & (heat_index_f < config.HEAT_UNSAFE_F)
        & (aqi < config.AQI_UNSAFE)
    )


def recommend(
    panel: pd.DataFrame, risk_col: str = "risk", window: int = config.SHIFT_WINDOW_H
) -> pd.DataFrame:
    """Per active hour, choose keep or the lowest-risk safe time shift."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a nonnegative integer")
    df = panel.copy()
    if df["ts_local"].isna().any():
        raise ValueError("recommendations require valid local timestamps")
    if not df["hour"].eq(df["ts_local"].dt.hour).all():
        raise ValueError("hour must match the local timestamp")
    probabilities = df[risk_col].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("risk must contain finite probabilities in [0, 1]")
    orig_index = df.index
    df = df.reset_index(drop=True)
    df["day"] = df["ts_local"].dt.date
    df["safe"] = _safe(df["heat_index_f"], df["aqi"])

    n = len(df)
    actions = np.empty(n, dtype=object)
    targets = np.full(n, np.nan)
    target_risk = np.empty(n)
    t_heat = np.empty(n)
    t_aqi = np.empty(n)

    for _, day in df.groupby("day"):
        idx = day.index.to_numpy()
        hours = day["hour"].to_numpy()
        risk = day[risk_col].to_numpy()
        safe = day["safe"].to_numpy()
        heat = day["heat_index_f"].to_numpy()
        air = day["aqi"].to_numpy()
        timestamps = day["ts_local"].to_numpy()
        for i in range(len(day)):
            pos = idx[i]
            near = np.abs(hours - hours[i]) <= window
            feasible = near & safe
            if not feasible.any():
                actions[pos] = "cancel"
                target_risk[pos] = 1.0
                t_heat[pos] = heat[i]
                t_aqi[pos] = air[i]
                continue
            cand = np.where(feasible)[0]
            order = np.lexsort(
                (timestamps[cand], np.abs(hours[cand] - hours[i]), risk[cand])
            )
            best = cand[order[0]]
            if not safe[i]:
                action = "shift"
            elif best == i or (risk[i] - risk[best]) < config.MIN_RISK_BENEFIT:
                action, best = "keep", i
            else:
                action = "shift"
            actions[pos] = action
            targets[pos] = hours[best]
            target_risk[pos] = risk[best]
            t_heat[pos] = heat[best]
            t_aqi[pos] = air[best]

    df["action"] = actions
    df["target_hour"] = targets
    df["chosen_risk"] = target_risk
    df["target_heat_index_f"] = t_heat
    df["target_aqi"] = t_aqi
    df.index = orig_index
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
