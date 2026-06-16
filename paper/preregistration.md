# Analysis plan and specification

This document fixes the primary specification and the line between confirmatory and exploratory
results. It is an honest specification lock, not a back-dated registration: a true preregistration
would be timestamped before any analysis. Its purpose is to make the analyst choices explicit and to
bind a future replication (the multi-city extension) to a single, declared spec.

## Research questions

- **RQ1 (estimation).** Does air quality suppress outdoor mobility within a day, once season and
  weather are held fixed?
- **RQ2 (forecasting).** Can hourly suppression be forecast in a *calibrated*, decision-usable way, and
  does calibration — not discrimination — govern usability?
- **RQ3 (policy).** How much otherwise-lost mobility can a safety-constrained time-shift recover without
  ever recommending an unsafe hour?

## Locked primary specification

- **Unit:** city-hour (Washington DC, 2022–2024).
- **Outcome / label:** an active hour (weather-free expected rides `E_t ≥ 20`) is *suppressed* when
  `rides_t < 0.5 · E_t`. `E_t` is the train-years median over `season × daytype × hour`, scaled by a
  volume level carried forward from the last training year (leak-free).
- **Primary causal estimand (RQ1):** the within-day fixed-effects coefficient of the hourly ride ratio
  on hourly AQI, with day and hour-of-day fixed effects and hourly weather controls, on the days with
  genuine intraday AQI variation. Reported per +50 AQI with a day-clustered bootstrap CI **and** the
  minimum detectable effect at 80%/90% power.
- **Primary forecast model (RQ2):** a standardized, unweighted logistic regression on the twelve locked
  features (heat index, two temperature hinges, hourly AQI, humidity, wind, precipitation, visibility,
  smoke flag, cyclical hour, weekend). Train 2022–2023, test 2024.
- **Inference:** 1,000-resample percentile bootstrap; day-clustered for hourly metrics and effects.

## Confirmatory (decided before looking at results)

1. Calibration of the served model vs the class-weighted variant (Brier, log loss, ECE, slope).
2. The within-day AQI effect and its power/MDE.
3. The feature ablation: marginal out-of-time value of the air-quality group.

## Exploratory (flagged as such; not for confirmatory claims)

- Recovered Active Minutes and the recovery percentage (a policy upper bound).
- The single June-2023 wildfire-smoke day.
- The between-day controlled regression (susceptible to residual day-level confounding).
- Equity by rider type, and the Seoul cross-city transfer.

## Pre-specified sensitivity sweeps

| Choice | Values |
| --- | --- |
| Label ratio ρ | 0.4, 0.5, 0.6 |
| Activity floor | 10, 20, 30 |
| Shift window (RAM) | ±2, ±3, ±4 h |
| High-AQI threshold | 80, 100, 120 |

A result is reported as robust only if it holds across its sweep.

## Replication plan (multi-city extension)

The same spec runs per city by changing the city block in `pulseshift/config.py` (coordinates, station,
trip source). Hourly AQI is available keyless for any city via CAMS/Open-Meteo, so the within-day
identification (RQ1) can be replicated wherever hourly trip data exists; the confounding result is
confirmed only if the between- vs within-day gap reproduces across cities.
