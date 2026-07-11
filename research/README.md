# PulseShift research pipeline

The analysis and model behind PulseShift: build an hourly dataset for Washington DC from public
sources, fit and evaluate a suppression-forecasting model, and export the model used by the app.

## Requirements

Python 3.12 or newer.

## Installation

From the repository root, `make all` builds the environment, runs the analysis, exports the model, and
runs the tests in one step. To set up manually:

```
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements.txt
```
## Usage

Run from the `research/` directory:

```
PYTHONPATH=. ../.venv/bin/python scripts/build_data.py     # download sources, build the panel
PYTHONPATH=. ../.venv/bin/python scripts/run_analysis.py   # write figures and tables to ../paper
PYTHONPATH=. ../.venv/bin/python scripts/train_model.py    # export ../model.js and ../model.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_seoul.py # cross-city check (downloads Seoul data; needs network)
PYTHONPATH=. ../.venv/bin/pytest tests/ -q                 # tests
```

The processed panel (`data/processed/panel.csv.gz`) is committed, so analysis and tests run without
re-downloading the raw data.

## Configuration

Study years, weather station, county code, and the label and safety thresholds are defined in
`pulseshift/config.py`. No API keys are required.

## Data

All sources are public and downloaded by the pipeline; nothing is hand-edited.

| Source | Provides |
| --- | --- |
| Capital Bikeshare trip history | Hourly outdoor-activity volume |
| NOAA Local Climatological Data (Reagan National) | Hourly temperature, humidity, wind, visibility, precipitation, smoke/haze |
| CAMS hourly air quality (via Open-Meteo, no key) | Hourly US AQI and PM2.5 |
| EPA AirData daily AQI (District of Columbia) | Daily air-quality anchor and smoke cross-check |

Activity (DST-aware local time) and weather (local standard time) are aligned on UTC; daily AQI is
joined on the local calendar day. The suppression outcome is constructed, since no public dataset
records skipped sessions: an active hour is suppressed when ridership falls below half of a
weather-free `season x daytype x hour` climatology, fit on the training years only.

## Layout

| Path | Purpose |
| --- | --- |
| `pulseshift/ingest.py` | Download and assemble the source series |
| `pulseshift/panel.py` | Merge, label suppression, write the analysis panel |
| `pulseshift/features.py` | Heat index, temperature hinges, temporal encodings |
| `pulseshift/models.py` | Climatology baseline, logistic, and gradient-boosting models |
| `pulseshift/calibration.py` | Probability calibration and reliability |
| `pulseshift/evaluation.py` | Discrimination and calibration metrics, bootstrap CIs |
| `pulseshift/airquality.py` | Air-quality identification ladder (marginal, between-day, within-day) |
| `pulseshift/ram.py` | Time-shift adaptation and Recovered Active Minutes |
| `pulseshift/decision.py` | Decision-curve net benefit |
| `pulseshift/equity.py` | Subgroup stratification |
| `pulseshift/safety.py` | Audit that recommendations never raise exposure |
| `pulseshift/plots.py` | Figures |
| `scripts/` | Pipeline entry points |
| `tests/` | Test suite |

## License

MIT; see the repository root.
