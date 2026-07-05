"""Assemble the hourly analysis panel and define the suppression label."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config, ingest
from .features import add_temporal, heat_index_f


def expected_rides(df: pd.DataFrame, value: str = "rides_total") -> np.ndarray:
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


def suppression_mask(
    rides: np.ndarray,
    expected: np.ndarray,
    ratio: float = config.SUPPRESSION_RATIO,
    floor: float = config.EXPECTED_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Active and suppressed boolean masks; the one rule every call site shares."""
    active = expected >= floor
    suppressed = (rides < ratio * expected) & active
    return active, suppressed


def label_suppression(
    df: pd.DataFrame,
    ratio: float = config.SUPPRESSION_RATIO,
    floor: float = config.EXPECTED_FLOOR,
) -> tuple[pd.Series, pd.Series]:
    active, suppressed = suppression_mask(
        df["rides_total"], df["expected_rides"], ratio, floor
    )
    return active, suppressed.astype(int)


PANEL_PATH = config.PROCESSED / "panel.csv.gz"
CHECKSUM_PATH = config.PROCESSED / "panel.sha256"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksum(path=PANEL_PATH, checksum_path=CHECKSUM_PATH) -> None:
    """Raise if the committed panel no longer matches its recorded hash."""
    if not checksum_path.exists():
        return
    expected = checksum_path.read_text().split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"panel checksum mismatch: {actual} != {expected}")


def build_panel(write: bool = True) -> pd.DataFrame:
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
    panel["cold_stress"] = (config.COLD_STRESS_BASE_F - panel["temp_f"]).clip(lower=0)
    panel["heat_stress"] = (panel["heat_index_f"] - config.HEAT_STRESS_BASE_F).clip(
        lower=0
    )
    panel = panel.dropna(subset=["temp_f", "humidity", "aqi"]).reset_index(drop=True)

    panel["expected_rides"] = expected_rides(panel)
    panel["active_hour"], panel["suppressed"] = label_suppression(panel)

    if write:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        panel.to_csv(PANEL_PATH, index=False)
        CHECKSUM_PATH.write_text(f"{_sha256(PANEL_PATH)}  panel.csv.gz\n")
    return panel


def load_panel(verify: bool = True) -> pd.DataFrame:
    if not PANEL_PATH.exists():
        return build_panel()
    if verify:
        verify_checksum()
    return pd.read_csv(PANEL_PATH, parse_dates=["ts_utc", "ts_local"])


def active(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["active_hour"]].reset_index(drop=True)
