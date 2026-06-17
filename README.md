# PulseShift

PulseShift predicts whether weather or air quality will ruin a planned outdoor session, and suggests the safest time to do it anyway. It's a single web page that runs the model right in your browser, with a Python pipeline and a full write-up behind it, built on three years of real Washington DC data.

Live demo: https://urme-b.github.io/PulseShift/

## What it does

Give it the conditions for your session and it returns the odds the weather will cost you, plus a recommendation. One rule always overrides the model: if the heat index hits 103°F or the AQI hits 150, it's flagged unsafe, full stop. And if you're on the live forecast, it also points out the safest hour of the day to go.

## Results

Washington DC, 2022–2024. The metrics come from a leak-free out-of-time split (train on 2022–2023, test on 2024), and the model the app actually serves is then refit on all three years.

What makes a forecast usable isn't raw accuracy, it's calibration. Class-weighting the model leaves it badly overconfident; the unweighted version the app serves stays on the diagonal.

<img src="paper/figures/reliability.png" alt="Reliability diagram: served model tracks the ideal line; the balanced model is overconfident" width="440">

The headline finding is about air quality, and it's a cautionary one: the effect of pollution on this mobility proxy depends entirely on how you measure it. The naive association looks *protective*. A season-controlled daily regression finds a large negative effect. But a within-day fixed-effects estimate, which holds season and weather fixed, collapses it toward zero. That gap is seasonal confounding, and only hourly air quality exposes it.

<img src="paper/figures/aqi_identification.png" alt="Forest plot: the air-quality effect shrinks from a large negative value to near zero as identification tightens" width="560">

- Features push forecast AUROC from 0.69 (a season-aware baseline) up to 0.94, and gradient boosting doesn't do any better.
- The served model is well calibrated out-of-time (Brier 0.021, ECE 0.044).
- **Air quality adds nothing to the forecast beyond weather** (ΔAUROC ≈ 0). Here it's cold and rain that drive suppression, not smoke.
- That daily air-quality effect (−10.9 ride-ratio points per +50 AQI) shrinks to −2.4 (95% CI −5.9 to +1.1) once you identify it within-day.
- A safety-constrained time-shift policy claws back up to ~37% of otherwise-lost activity (an upper bound), and never recommends an unsafe hour.
- The same pipeline reaches AUROC 0.87 on a second city, Seoul.

The full write-up, figures, and confidence intervals are in [`paper/paper.md`](paper/paper.md).

## How it works

A logistic regression predicts whether a given hour of outdoor cycling is suppressed, meaning ridership drops below half of a weather-free seasonal and diurnal baseline. The features are heat index, hourly AQI, humidity, wind, precipitation, two temperature hinge terms, visibility, a smoke flag, and a cyclical encoding of hour and weekend. Training happens in Python; the model is exported to `model.js` as twelve coefficients, and the page standardizes the inputs and applies them directly. Inference is just a dot product through a sigmoid, so there's no server, no build step, and no API keys.

## Method

- **Leak-free, out-of-time validation.** Trained on 2022–2023, tested on a held-out 2024. The suppression label uses a climatology fit on the training years only, with the test-year volume carried forward, so nothing from the test period leaks into the label.
- **Hourly air quality.** Daily AQI is swapped for hourly AQI (CAMS reanalysis, anchored to EPA daily ground-station data), which is what makes within-day identification possible in the first place.
- **An identification ladder for air quality.** The AQI effect is estimated three ways: marginally, with a season-controlled between-day regression, and with a within-day fixed-effects model that absorbs every day-level confounder. Most of the apparent effect turns out to be seasonal confounding.
- **A feature ablation.** Temporal, then weather, then air quality, out-of-time, to isolate what air quality actually adds to the forecast (essentially nothing).
- **Calibration as a first-class metric.** Brier score, log loss, expected calibration error, calibration slope, and a decision curve, compared across unweighted, class-weighted, and isotonic-recalibrated variants, not AUROC alone.
- **Quantified uncertainty.** 95% confidence intervals by bootstrap, day-clustered for the within-day estimator and resampled over days for the policy metric.
- **Power, not just a null.** The within-day design reports its minimum detectable effect, so the near-zero air-quality result is a bounded-small effect rather than an underpowered shrug.
- **Sensitivity over silence.** Every analyst choice — label ratio, activity floor, shift window, AQI threshold — is swept and reported, with the spec and the confirmatory/exploratory split fixed in [`paper/preregistration.md`](paper/preregistration.md).

The full pipeline, paper, and tests live in [`research/`](research/) and [`paper/`](paper/).

## Tech stack

| Layer | Choice |
| --- | --- |
| App | Vanilla HTML, CSS, and JavaScript — zero dependencies, static-hostable |
| Model | scikit-learn logistic regression, exported to JSON and run in the browser |
| Pipeline | Python, pandas, NumPy, scikit-learn, matplotlib |
| Data | Capital Bikeshare, NOAA Local Climatological Data, EPA AirData, CAMS hourly air quality (all public) |
| Quality | pytest, GitHub Actions CI, bootstrap confidence intervals, decision-curve analysis |

## Reproduce

```
make all       # venv, analysis, model export, tests
```

Requires Python 3.9–3.12 (all tested in CI). Or step by step, from `research/`: see [`research/README.md`](research/README.md). The processed panel is committed (with a checksum and a column-level [data dictionary](research/data/README.md)), so analysis and tests run without re-downloading the raw data.

## Future scope

- [ ] Validate the ridership proxy against measured activity (wearable or survey data)
- [ ] Replace CAMS hourly AQI with ground-station hourly measurements
- [ ] Replicate the confounding result across cities (the pipeline is city-configurable)
- [ ] Neighborhood-level equity analysis
- [ ] Expose the forecast as a small API

## Contributing

Setup, test, and reproduction steps are in [`CONTRIBUTING.md`](CONTRIBUTING.md); participation is
governed by the [Code of Conduct](CODE_OF_CONDUCT.md), and vulnerabilities go through the
[security policy](SECURITY.md).

## License

[MIT License](LICENSE) © 2026 Urme Bose
