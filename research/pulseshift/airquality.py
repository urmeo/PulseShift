"""Air-quality identification: marginal, between-day, and within-day effects."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

CONTROLS = ["temp_f", "precip_in", "wind_mph", "humidity"]


def _ride_ratio(work):
    df = work.assign(
        day=work["ts_local"].dt.normalize(),
        ride_ratio=work["rides_total"] / work["expected_rides"],
    )
    return df[np.isfinite(df["ride_ratio"])].reset_index(drop=True)


def between_day_effect(work, controls=CONTROLS, n_boot=1000, seed=0):
    """Daily ride ratio on daily-peak AQI, controlling for weather and season."""
    df = _ride_ratio(work)
    g = df.groupby("day").agg(
        ride_ratio=("rides_total", "sum"),
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
    y = (g["ride_ratio"] / g["expected"]).to_numpy()
    X = pd.concat(
        [
            g[["aqi", *controls, "is_weekend"]],
            pd.get_dummies(g["season"], prefix="s", drop_first=True),
        ],
        axis=1,
    ).astype(float)
    return _ols_ci(X.to_numpy(), y, 0, len(g), n_boot, seed)


def within_day_effect(work, controls=CONTROLS, n_boot=1000, seed=0):
    """Hourly ride ratio on AQI with day and hour fixed effects (intraday)."""
    df = _ride_ratio(work)
    varies = (
        df.groupby("day")["aqi"].transform("nunique") > 1
    )  # drop flat-fallback days
    df = df[varies].reset_index(drop=True)
    reg = ["aqi", *controls]
    hours = pd.get_dummies(df["hour"], prefix="h", drop_first=True).astype(float)
    block = pd.concat([df[["ride_ratio", *reg]], hours], axis=1)
    demeaned = block - block.groupby(df["day"].to_numpy()).transform("mean")

    y = demeaned["ride_ratio"].to_numpy()
    X = demeaned.drop(columns="ride_ratio").to_numpy()
    days = df["day"].to_numpy()
    return _ols_ci(X, y, 0, df["day"].nunique(), n_boot, seed, cluster=days)


def _ols_ci(X, y, k, n_unit, n_boot, seed, cluster=None, scale=50):
    """Coefficient k (scaled per +50 AQI) with a percentile bootstrap CI."""
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
    lo, hi = np.percentile(np.array(vals) * scale, [2.5, 97.5])
    return {
        "effect_per_50": round(point, 3),
        "ci_low": round(float(lo), 3),
        "ci_high": round(float(hi), 3),
        "n": int(n_unit),
    }


def smoke_episodes(work, aqi_thresh=100):
    """Ride ratio on high-AQI hours vs same season-hour clean baseline."""
    df = _ride_ratio(work)
    df["polluted"] = df["aqi"] >= aqi_thresh
    base = (
        df[~df["polluted"]]
        .groupby(["season", "hour"])["ride_ratio"]
        .mean()
        .rename("clean_ratio")
    )
    hot = df[df["polluted"]].merge(base, on=["season", "hour"], how="left")
    hot["rel"] = hot["ride_ratio"] / hot["clean_ratio"]
    return {
        "aqi_threshold": aqi_thresh,
        "polluted_hours": int(len(hot)),
        "polluted_days": int(df.loc[df["polluted"], "day"].nunique()),
        "ride_ratio_vs_clean": round(float(hot["rel"].mean()), 3),
        "median_aqi_polluted": round(float(df.loc[df["polluted"], "aqi"].median()), 1),
    }
