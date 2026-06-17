# Data

Three public sources for Washington DC, 2022–2024, joined to one hourly panel.

| Stage | Path | Tracked in git? |
| --- | --- | --- |
| Raw downloads | `data/raw/` | No (gitignored) |
| Per-source interim caches | `data/interim/` | No (gitignored) |
| Processed analysis panel | `data/processed/panel.csv.gz` | **Yes** |

Only the processed panel is committed, so analysis and tests run offline. `make data` rebuilds it
from the public sources (requires network); `ingest.py` and `panel.py` are the build code.

## Integrity

The committed panel has a recorded checksum. Verify it before trusting a local copy:

```bash
cd research/data/processed && shasum -a 256 -c panel.sha256   # expects: panel.csv.gz: OK
```

## Provenance

- **Activity** — Capital Bikeshare trip files, aggregated to hourly counts by rider type.
- **Weather** — NOAA Local Climatological Data, Reagan National (DCA, WBAN 13743).
- **Air quality** — hourly US AQI / PM2.5 from CAMS reanalysis via Open-Meteo, anchored to EPA
  AirData daily AQI. CAMS hourly coverage begins August 2022, so `aqi_hourly` is present for ~80%
  of city-hours; earlier hours fall back to the EPA daily value.

## Scope note

`panel.csv.gz` holds all 26,288 city-hours. The forecasting analysis runs on the **active** subset
(`active_hour == True`): 24,354 hours, split 16,150 (2022–2023, train) / 8,204 (2024, test). The
within-day air-quality estimator uses the 881 of 1,096 days that carry genuine intraday AQI variation.

`pm25` (raw PM2.5) is provided for downstream reuse and reproducibility; the model and analysis use the
composite `aqi`, so `pm25` is published but not consumed here.

## Columns (31)

| Column | Type | Unit / values | Description | Source |
| --- | --- | --- | --- | --- |
| `ts_utc` | datetime | UTC hour start | Join key | derived |
| `rides_member` | int | count | Member rides in the hour | Capital Bikeshare |
| `rides_casual` | int | count | Casual rides in the hour | Capital Bikeshare |
| `rides_total` | int | count | `rides_member + rides_casual` | derived |
| `temp_f` | float | °F | Dry-bulb temperature | NOAA LCD |
| `humidity` | float | % (0–100) | Relative humidity | NOAA LCD |
| `wind_mph` | float | mph | Wind speed | NOAA LCD |
| `visibility_mi` | float | miles | Visibility | NOAA LCD |
| `precip_in` | float | inches | Hourly precipitation (trace → 0) | NOAA LCD |
| `smoke_haze` | int | {0,1} | Present-weather code HZ/FU/smoke/haze | NOAA LCD |
| `ts_local` | datetime | America/New_York | Local wall-clock hour | derived |
| `date` | date | local day | Local calendar day | derived |
| `aqi_epa_daily` | int | US AQI | Authoritative daily AQI (ffill/bfill) | EPA AirData |
| `aqi_hourly` | float | US AQI | Hourly AQI (CAMS; ~80% coverage) | CAMS / Open-Meteo |
| `pm25` | float | µg/m³ | Hourly PM2.5 | CAMS / Open-Meteo |
| `aqi` | float | US AQI | Hourly where present, else daily fallback (model input) | derived |
| `hour` | int | 0–23 | Local hour | derived |
| `dow` | int | 0=Mon … 6=Sun | Local day of week | derived |
| `month` | int | 1–12 | Local month | derived |
| `year` | int | 2022–2024 | Local year | derived |
| `daytype` | str | {weekday, weekend} | Day type | derived |
| `season` | str | {winter, spring, summer, fall} | Meteorological season | derived |
| `hour_sin` | float | [-1, 1] | Cyclical hour encoding | derived |
| `hour_cos` | float | [-1, 1] | Cyclical hour encoding | derived |
| `is_weekend` | int | {0,1} | Weekend flag | derived |
| `heat_index_f` | float | °F | NWS Rothfusz heat index | derived |
| `cold_stress` | float | °F | `max(0, 55 − temp_f)` | derived |
| `heat_stress` | float | °F | `max(0, heat_index_f − 85)` | derived |
| `expected_rides` | float | count | Weather-free climatology baseline `E_t` (train-fit, leak-free) | derived |
| `active_hour` | bool | — | `E_t ≥ 20` (activity floor) | derived |
| `suppressed` | int | {0,1} | `rides_total < 0.5 · E_t` and active — the label | derived |
