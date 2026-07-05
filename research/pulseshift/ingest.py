"""Download and assemble real DC activity, weather, and air-quality series."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def _download(url: str, dest) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pulseshift-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)  # atomic: an interrupted download never looks complete
    return dest


def _write_csv(df: pd.DataFrame, cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(cache)  # atomic: a crash mid-write never leaves a truncated cache


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_bikeshare() -> pd.DataFrame:
    """City-wide hourly ride counts (UTC), split by rider type."""
    cache = config.INTERIM / "bikeshare_hourly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["ts_utc"])

    frames = []
    for ym in config.ym_list():
        path = _download(
            config.BIKESHARE_URL.format(ym=ym), config.RAW / f"cabi_{ym}.zip"
        )
        with zipfile.ZipFile(path) as archive:
            members = [
                n
                for n in archive.namelist()
                if n.lower().endswith(".csv") and "macosx" not in n.lower()
            ]
            for name in members:
                with archive.open(name) as handle:
                    df = pd.read_csv(
                        handle,
                        usecols=["started_at", "member_casual"],
                        low_memory=False,
                    )
                frames.append(df)

    trips = pd.concat(frames, ignore_index=True)
    started = pd.to_datetime(trips["started_at"], format="mixed", errors="coerce")
    local = started.dt.tz_localize(
        config.LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward"
    )
    trips = trips.assign(ts_utc=local.dt.tz_convert("UTC").dt.floor("h")).dropna(
        subset=["ts_utc"]
    )
    trips["ts_utc"] = trips["ts_utc"].dt.tz_localize(None)

    trips = trips.dropna(subset=["member_casual"])
    pivot = trips.pivot_table(
        index="ts_utc", columns="member_casual", aggfunc="size", fill_value=0
    ).rename(columns={"member": "rides_member", "casual": "rides_casual"})
    pivot["rides_total"] = pivot.sum(axis=1)
    out = pivot.reset_index()[["ts_utc", "rides_member", "rides_casual", "rides_total"]]
    _write_csv(out, cache)
    return out


def load_weather() -> pd.DataFrame:
    """Hourly DCA weather (UTC) from NOAA LCD."""
    cache = config.INTERIM / "weather_hourly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["ts_utc"])

    cols = [
        "DATE",
        "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity",
        "HourlyWindSpeed",
        "HourlyVisibility",
        "HourlyPrecipitation",
        "HourlyPresentWeatherType",
    ]
    frames = []
    for year in config.YEARS:
        path = _download(
            config.LCD_URL.format(year=year, station=config.LCD_STATION),
            config.RAW / f"lcd_{year}.csv",
        )
        df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
        frames.append(df)

    lcd = pd.concat(frames, ignore_index=True)
    stamp = pd.to_datetime(lcd["DATE"], errors="coerce")
    # LCD is Local Standard Time (no DST); rides use wall-clock local.
    # Both resolve to true UTC, so the hourly join pairs simultaneous conditions.
    lcd["ts_utc"] = (
        stamp.dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC").dt.tz_localize(None)
    )
    lcd["smoke_haze"] = (
        lcd["HourlyPresentWeatherType"]
        .astype(str)
        .str.contains("HZ|FU|smoke|haze", case=False, na=False)
    ).astype(int)
    for col in [
        "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity",
        "HourlyWindSpeed",
        "HourlyVisibility",
    ]:
        lcd[col] = _to_numeric(lcd[col])
    # precipitation: trace/blank read as 0
    lcd["HourlyPrecipitation"] = _to_numeric(lcd["HourlyPrecipitation"]).fillna(0)

    lcd["hour"] = lcd["ts_utc"].dt.floor("h")
    hourly = (
        lcd.dropna(subset=["hour"])
        .groupby("hour")
        .agg(
            temp_f=("HourlyDryBulbTemperature", "mean"),
            humidity=("HourlyRelativeHumidity", "mean"),
            wind_mph=("HourlyWindSpeed", "mean"),
            visibility_mi=("HourlyVisibility", "mean"),
            precip_in=("HourlyPrecipitation", "max"),
            smoke_haze=("smoke_haze", "max"),
        )
        .reset_index()
        .rename(columns={"hour": "ts_utc"})
    )
    _write_csv(hourly, cache)
    return hourly


def load_aqi() -> pd.DataFrame:
    """Daily DC AQI from EPA AirData."""
    cache = config.INTERIM / "aqi_daily.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    use = ["State Name", "Date", "AQI"]
    parts = []
    for year in config.YEARS:
        path = _download(
            config.EPA_DAILY_AQI_URL.format(year=year),
            config.RAW / f"epa_aqi_{year}.zip",
        )
        df = pd.read_csv(path, usecols=use, compression="zip")
        parts.append(df[df["State Name"] == "District Of Columbia"])

    aqi = pd.concat(parts, ignore_index=True)
    daily = (
        aqi.assign(date=pd.to_datetime(aqi["Date"]))
        .groupby("date")
        .agg(aqi=("AQI", "max"))
        .reset_index()
    )
    _write_csv(daily, cache)
    return daily


def load_aqi_hourly() -> pd.DataFrame:
    """Hourly US AQI and PM2.5 for DC (CAMS reanalysis, local time)."""
    cache = config.INTERIM / "aqi_hourly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["ts_local"])

    frames = []
    for year in config.YEARS:
        url = config.OPENMETEO_AQI_URL.format(
            lat=config.DC_LAT, lon=config.DC_LON, year=year
        )
        dest = _download(url, config.RAW / f"aqi_hourly_{year}.json")
        h = json.loads(Path(dest).read_text())["hourly"]
        frames.append(
            pd.DataFrame(
                {"ts_local": h["time"], "aqi_hourly": h["us_aqi"], "pm25": h["pm2_5"]}
            )
        )

    hourly = pd.concat(frames, ignore_index=True)
    hourly["ts_local"] = pd.to_datetime(hourly["ts_local"])
    _write_csv(hourly, cache)
    return hourly
