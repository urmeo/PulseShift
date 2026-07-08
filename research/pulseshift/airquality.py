"""Air-quality identification: marginal, between-day, within-day, and power."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config

CONTROLS = ["temp_f", "precip_in", "wind_mph", "humidity"]

Z80, Z90 = config.MDE_Z80, config.MDE_Z90


def _ride_ratio(work: pd.DataFrame) -> pd.DataFrame:
    df = work.assign(
        day=work["ts_local"].dt.normalize(),
        ride_ratio=work["rides_total"] / work["expected_rides"],
    )
    return df[np.isfinite(df["ride_ratio"])].reset_index(drop=True)


def between_day_effect(
    work: pd.DataFrame,
    controls: list[str] = CONTROLS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Daily ride ratio on daily-peak AQI, controlling for weather and season."""
    df = _ride_ratio(work)
    g = df.groupby("day").agg(
        rides=("rides_total", "sum"),
        expected=("expected_rides", "sum"),
        aqi=("aqi", "max"),
        temp_f=("temp_f", "mean"),
        precip_in=("precip_in", "sum"),
        wind_mph=("wind_mph", "mean"),
        humidity=("humidity", "mean"),
        is_weekend=("is_weekend", "max"),
        season=("season", "first"),
    )
    g = g[g["expected"] > 0]
    y = (g["rides"] / g["expected"]).to_numpy()
    X = pd.concat(
        [
            g[["aqi", *controls, "is_weekend"]],
            pd.get_dummies(g["season"], prefix="s", drop_first=True),
        ],
        axis=1,
    ).astype(float)
    return _ols_ci(X.to_numpy(), y, 0, len(g), n_boot, seed)


def within_day_effect(
    work: pd.DataFrame,
    controls: list[str] = CONTROLS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Hourly ride ratio on AQI with day and hour fixed effects (intraday)."""
    df = _ride_ratio(work)
    varies = df.groupby("day")["aqi"].transform("nunique") > 1  # drop flat days
    df = df[varies].reset_index(drop=True)
    reg = ["aqi", *controls]
    hours = pd.get_dummies(df["hour"], prefix="h", drop_first=True).astype(float)
    block = pd.concat([df[["ride_ratio", *reg]], hours], axis=1)
    demeaned = block - block.groupby(df["day"].to_numpy()).transform("mean")

    y = demeaned["ride_ratio"].to_numpy()
    X = demeaned.drop(columns="ride_ratio").to_numpy()
    days = df["day"].to_numpy()
    return _ols_ci(X, y, 0, df["day"].nunique(), n_boot, seed, cluster=days)


def _ols_ci(
    X, y, k: int, n_unit: int, n_boot: int, seed: int, cluster=None, scale: int = 50
) -> dict:
    """Coefficient k per +50 AQI, with bootstrap CI, SE, and detectable effect."""
    point = float(LinearRegression().fit(X, y).coef_[k] * scale)
    rng = np.random.default_rng(seed)
    vals = []
    if cluster is None:
        idx = np.arange(len(y))
        for _ in range(n_boot):
            s = rng.choice(idx, len(idx), replace=True)
            vals.append(LinearRegression().fit(X[s], y[s]).coef_[k])
    else:
        groups = {g: np.where(cluster == g)[0] for g in np.unique(cluster)}
        keys = list(groups)
        for _ in range(n_boot):
            s = np.concatenate(
                [groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
            )
            vals.append(LinearRegression().fit(X[s], y[s]).coef_[k])
    arr = np.array(vals) * scale
    lo, hi = np.percentile(arr, [2.5, 97.5])
    se = float(arr.std(ddof=1))
    return {
        "effect_per_50": round(point, 3),
        "ci_low": round(float(lo), 3),
        "ci_high": round(float(hi), 3),
        "se": round(se, 3),
        "mde80": round(Z80 * se, 3),
        "mde90": round(Z90 * se, 3),
        "n": int(n_unit),
    }


def measurement_error_bound(
    work: pd.DataFrame,
    beta_per_50: float,
    reliabilities: tuple[float, ...] = (0.7, 0.5, 0.3),
) -> dict:
    """Attenuation-corrected within-day effect under classical exposure error.

    CAMS reliability is the slope of EPA-daily on CAMS-daily (two same-scale
    error-prone measures of true AQI); corrected effect = beta / reliability.

    Caveat: this reliability is estimated at the DAILY level, but it corrects a
    WITHIN-DAY (intraday) estimate. Intraday CAMS reliability is unobserved and
    plausibly lower than daily, so the empirical row is conservative-correct only
    under the assumption that intraday reliability >= daily; the assumed-rho rows
    bracket the lower-reliability case. Ground-station hourly data would replace
    this assumption with a measured intraday reliability.
    """
    df = work.dropna(subset=["aqi_hourly"]).copy()
    df["day"] = df["ts_local"].dt.normalize()
    daily = (
        df.groupby("day")
        .agg(cams=("aqi_hourly", "mean"), epa=("aqi_epa_daily", "first"))
        .dropna()
    )
    cov = np.cov(daily["cams"], daily["epa"])
    lam = float(cov[0, 1] / cov[0, 0])
    r = float(np.corrcoef(daily["cams"], daily["epa"])[0, 1])
    rows = [
        {
            "reliability": "empirical",
            "rho": round(lam, 2),
            "corrected_per_50": round(beta_per_50 / lam, 3),
        }
    ]
    rows += [
        {
            "reliability": "assumed",
            "rho": rho,
            "corrected_per_50": round(beta_per_50 / rho, 3),
        }
        for rho in reliabilities
    ]
    return {"cams_epa_corr": round(r, 3), "reliability": round(lam, 3), "rows": rows}


def smoke_episodes(
    work: pd.DataFrame, aqi_thresh: int = 100, n_boot: int = 1000, seed: int = 0
) -> dict:
    """High-AQI hours vs same season-hour clean baseline, day-clustered CI."""
    df = _ride_ratio(work)
    df["polluted"] = df["aqi"] >= aqi_thresh
    base = (
        df[~df["polluted"]]
        .groupby(["season", "hour"])["ride_ratio"]
        .mean()
        .rename("clean_ratio")
    )
    hot = (
        df[df["polluted"]]
        .merge(base, on=["season", "hour"], how="left")
        .dropna(subset=["clean_ratio"])
    )
    rel = (hot["ride_ratio"] / hot["clean_ratio"]).to_numpy()

    rng = np.random.default_rng(seed)
    days = hot["day"].to_numpy()
    members = [np.where(days == d)[0] for d in np.unique(days)]
    boot = [
        rel[
            np.concatenate(
                [members[i] for i in rng.integers(0, len(members), len(members))]
            )
        ].mean()
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "aqi_threshold": aqi_thresh,
        "polluted_hours": len(hot),
        "polluted_days": int(hot["day"].nunique()),
        "ride_ratio_vs_clean": round(float(rel.mean()), 3),
        "ci_low": round(float(lo), 3),
        "ci_high": round(float(hi), 3),
        "median_aqi_polluted": round(
            float(hot["aqi"].median()), 1
        ),  # matched set, as counted
    }
