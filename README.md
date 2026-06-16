# PulseShift

PulseShift forecasts whether weather and air quality will suppress a planned outdoor session, and
recommends the safest way to keep it. It is a single static web page that runs a calibrated model in
the browser, backed by a reproducible Python pipeline and a written analysis on three years of real
Washington DC data.

Live demo: https://urme-b.github.io/PulseShift/

## What it does

You give it the conditions for a planned session; it returns a calibrated probability that the session
will be lost to the weather, plus a recommendation. A hard safety rule sits above the model: a heat
index of 103°F or higher, or an AQI of 150 or higher, is always flagged unsafe. With the live forecast
it also reports the lowest-risk safe hour for the day.

## Results

Trained on Washington DC, 2022–2024, and evaluated out-of-time on a held-out 2024.

Calibration, not raw accuracy, is what makes a forecast usable. Class-weighting the model leaves it
badly overconfident; the unweighted model the app serves stays on the diagonal.

<img src="paper/figures/reliability.png" alt="Reliability diagram: served model tracks the ideal line; the balanced model is overconfident" width="440">

The headline finding is about air quality, and it is a cautionary one. The effect of pollution on
activity depends entirely on how you measure it. The naive association is *protective*; a
season-controlled daily regression finds a large negative effect; but a within-day fixed-effects
estimate — which holds season and weather fixed — collapses it toward zero. The gap is seasonal
confounding, and only hourly air quality exposes it.

<img src="paper/figures/aqi_identification.png" alt="Forest plot: the air-quality effect shrinks from a large negative value to near zero as identification tightens" width="560">

- Features lift forecast AUROC from 0.69 (a season-aware baseline) to 0.94; gradient boosting does not beat it.
- The served model is well calibrated out-of-time (Brier 0.021, ECE 0.044).
- **Air quality adds nothing to the forecast beyond weather** (ΔAUROC ≈ 0): cold and rain, not smoke, dominate suppression here.
- The daily air-quality effect (−10.9 ride-ratio points per +50 AQI) collapses to −2.3 (95% CI −5.5 to +1.0) under within-day identification.
- A safety-constrained time-shift policy recovers up to ~37% of otherwise-lost activity (an upper bound) while never recommending an unsafe hour.
- The same pipeline reaches AUROC 0.87 on a second city, Seoul.

Full write-up, figures, and confidence intervals: [`paper/paper.md`](paper/paper.md).

## How it works

A logistic regression predicts whether an hour of outdoor cycling is suppressed — ridership below half
of a weather-free seasonal and diurnal baseline. Features are heat index, hourly AQI, humidity, wind,
precipitation, two temperature hinge terms, visibility, a smoke flag, and a cyclical encoding of hour
and weekend. The model is trained in Python and exported to `model.js` as twelve coefficients; the page
standardizes the inputs and applies them directly, so inference is a dot product through a sigmoid — no
server, no build step, no keys.

## Method

- **Leak-free, out-of-time validation.** Trained on 2022–2023 and tested on a held-out 2024. The
  suppression label uses a climatology fit on the training years only, with the test-year volume carried
  forward, so no test-period information enters the label.
- **Hourly air quality.** Daily AQI is replaced by hourly AQI (CAMS reanalysis, anchored to EPA daily
  ground-station data), which is what makes within-day identification possible.
- **An identification ladder for air quality.** The AQI effect is estimated marginally, with a
  season-controlled between-day regression, and with a within-day fixed-effects model that absorbs every
  day-level confounder — showing that most of the apparent effect is seasonal confounding.
- **A feature ablation.** Temporal → weather → air quality, out-of-time, isolating what air quality
  actually contributes to the forecast (essentially nothing).
- **Calibration as a first-class metric.** Brier score, log loss, expected calibration error,
  calibration slope, and a decision curve, compared across unweighted, class-weighted, and
  isotonic-recalibrated variants — not AUROC alone.
- **Quantified uncertainty.** 95% confidence intervals by bootstrap, day-clustered for the within-day
  estimator and resampled over days for the policy metric.

The full pipeline, paper, and tests live in [`research/`](research/) and [`paper/`](paper/).

## Tech stack

| Layer | Choice |
| --- | --- |
| App | Vanilla HTML, CSS, and JavaScript — zero dependencies, static-hostable |
| Model | scikit-learn logistic regression, exported to JSON and run in the browser |
| Pipeline | Python, pandas, scikit-learn, scipy, matplotlib |
| Data | Capital Bikeshare, NOAA Local Climatological Data, EPA AirData, CAMS hourly air quality (all public) |
| Quality | pytest, GitHub Actions CI, bootstrap confidence intervals, decision-curve analysis |

## Reproduce

```
make all       # venv, panel, analysis, model export, tests
```

Or step by step, from `research/`: see [`research/README.md`](research/README.md). The processed panel
is committed, so analysis and tests run without re-downloading the raw data.

## Future scope

- [ ] Validate against observed session data, not a ridership proxy
- [ ] Replace CAMS hourly AQI with ground-station hourly measurements
- [ ] Extend to more cities and activities
- [ ] Neighborhood-level equity analysis
- [ ] Expose the forecast as a small API

## License

[MIT License](LICENSE)
