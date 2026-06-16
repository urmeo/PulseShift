"""Assemble the hourly analysis panel and define the suppression label."""

import pandas as pd

from . import config, ingest
from .features import add_temporal, heat_index_f


def _expected_rides(df, value="rides_total"):
    """Train-fit shape and volume level; test years carry the last train year (leak-free)."""
    keys = ["season", "daytype", "hour"]
    train = df[df["year"].isin(config.TRAIN_YEARS)]
    shape = (
        train.groupby(keys)[value]
        .median()
        .reindex(pd.MultiIndex.from_frame(df[keys]))
        .to_numpy()
    )
    year_level = train.groupby("year")[value].mean() / train[value].mean()
    last_train = year_level.loc[max(config.TRAIN_YEARS)]
    level = df["year"].map(lambda y: year_level.get(y, last_train)).to_numpy()
    return shape * level


def label_suppression(df, ratio=config.SUPPRESSION_RATIO, floor=config.EXPECTED_FLOOR):
    expected = df["expected_rides"]
    active = expected >= floor
    suppressed = (df["rides_total"] < ratio * expected) & active
    return active, suppressed.astype(int)


def build_panel(write=True):
    bikes = ingest.load_bikeshare()
    weather = ingest.load_weather()
    aqi = ingest.load_aqi()

    for frame in (bikes, weather):
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"])

    panel = (
        bikes.merge(weather, on="ts_utc", how="inner")
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )
    for col in ["temp_f", "humidity", "wind_mph", "visibility_mi"]:
        panel[col] = panel[col].interpolate(limit=3).ffill(limit=3).bfill(limit=3)

    panel["ts_local"] = (
        panel["ts_utc"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(config.LOCAL_TZ)
        .dt.tz_localize(None)
    )
    panel["date"] = panel["ts_local"].dt.normalize()
    daily = aqi[["date", "aqi"]].rename(columns={"aqi": "aqi_epa_daily"})
    panel = panel.merge(daily, on="date", how="left")
    panel["aqi_epa_daily"] = panel["aqi_epa_daily"].ffill().bfill()

    hourly = ingest.load_aqi_hourly()
    panel = panel.merge(hourly, on="ts_local", how="left")
    panel["aqi"] = panel["aqi_hourly"].fillna(panel["aqi_epa_daily"])
    panel["pm25"] = panel["pm25"].interpolate(limit=6)

    panel = add_temporal(panel)
    panel["is_weekend"] = (panel["daytype"] == "weekend").astype(int)
    panel["heat_index_f"] = heat_index_f(panel["temp_f"], panel["humidity"])
    panel["precip_in"] = panel["precip_in"].fillna(0)
    panel["cold_stress"] = (55.0 - panel["temp_f"]).clip(lower=0)
    panel["heat_stress"] = (panel["heat_index_f"] - 85.0).clip(lower=0)
    panel = panel.dropna(subset=["temp_f", "humidity", "aqi"]).reset_index(drop=True)

    panel["expected_rides"] = _expected_rides(panel)
    panel["active_hour"], panel["suppressed"] = label_suppression(panel)

    if write:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        panel.to_csv(config.PROCESSED / "panel.csv.gz", index=False)
    return panel


def load_panel():
    path = config.PROCESSED / "panel.csv.gz"
    if not path.exists():
        return build_panel()
    return pd.read_csv(path, parse_dates=["ts_utc", "ts_local"])


def active(panel):
    return panel[panel["active_hour"]].reset_index(drop=True)
