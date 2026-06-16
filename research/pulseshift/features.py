"""Feature construction: heat index, exposure bands, temporal encodings."""

import numpy as np
import pandas as pd


def heat_index_f(temp_f, humidity):
    """NWS Rothfusz heat index; identity below 80F / 40% RH."""
    t = np.asarray(temp_f, dtype=float)
    rh = np.asarray(humidity, dtype=float)
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 0.00683783 * t**2
        - 0.05481717 * rh**2
        + 0.00122874 * t**2 * rh
        + 0.00085282 * t * rh**2
        - 0.00000199 * t**2 * rh**2
    )
    mild = (t < 80) | (rh < 40)
    return np.where(mild, t, np.round(hi, 1))


def season_of(month):
    return (
        pd.cut(
            month,
            bins=[0, 2, 5, 8, 11, 12],
            labels=["winter", "spring", "summer", "fall", "winter2"],
            ordered=False,
        )
        .astype(str)
        .str.replace("winter2", "winter")
    )


def add_temporal(df, ts_col="ts_local"):
    ts = df[ts_col]
    out = df.copy()
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["year"] = ts.dt.year
    out["daytype"] = np.where(out["dow"] >= 5, "weekend", "weekday")
    out["season"] = season_of(out["month"])
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


MODEL_FEATURES = [
    "heat_index_f",
    "cold_stress",
    "heat_stress",
    "aqi",
    "humidity",
    "wind_mph",
    "precip_in",
    "visibility_mi",
    "smoke_haze",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]
