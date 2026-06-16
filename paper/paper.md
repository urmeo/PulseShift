# Does Air Quality Suppress Outdoor Activity? Seasonal Confounding, Hourly Identification, and a Calibrated Decision Tool

Urme B.

## Abstract

Heat, cold, rain, and wildfire smoke all disrupt outdoor physical activity, and air quality in
particular is a growing concern as wildfire smoke reaches cities that rarely saw it. But most studies
that link air quality to activity rely on *daily* pollution measures, and daily air quality is
strongly confounded by season: the dirtiest days are hot summer days, exactly when outdoor activity
peaks. We ask whether the air-quality association with outdoor activity — which we proxy with municipal
bike-share ridership, i.e. outdoor mobility — survives honest identification. Using three public sources
for Washington DC, 2022–2024 (Capital Bikeshare volumes, NOAA weather, and hourly air quality), we treat
the question as both an estimation problem and a calibrated forecasting problem.
We label each active city-hour suppressed when ridership falls below half of a weather-free, leak-free
climatology, and we replace daily AQI with hourly AQI so that pollution can be identified *within* a
day rather than across seasons. Three findings follow. First, the air-quality effect is an artifact of
how it is measured: the naive marginal association is *protective*, a season- and weather-controlled
between-day regression gives a large negative effect (−10.9 ride-ratio points per +50 AQI, 95% CI −14.6
to −7.3), but the rigorous within-day fixed-effects estimate collapses to −2.4 points (95% CI −5.9 to
+1.1), and high-AQI hours matched to clean same-season hours show no reduction (1.01×, 95% CI
0.94–1.09). This is not lost power: the within-day estimate is as precise as the between-day one (SE 1.8
vs 1.9) and would detect a between-day-sized effect with near-certainty, so the collapse reflects
de-confounding. Second, air
quality adds nothing to forecasting once weather is included (ΔAUROC and ΔAUPRC ≈ 0): weather is the
entire signal. Third, calibration, not discrimination, decides whether the forecast is usable:
class-weighting, a common default, leaves the model badly over-confident (Brier 0.13, ECE 0.25) while
the unweighted served model is well calibrated (Brier 0.021, ECE 0.044) at AUROC 0.94. A
safety-constrained time-shift policy recovers ~37% of otherwise-lost activity while never recommending
an unsafe hour, and the framework transfers to Seoul (AUROC 0.87). The lesson is that the headline
air-quality effect reported across this literature is largely seasonal confounding, and only hourly,
within-day identification corrects it.

**Keywords:** outdoor mobility; bike-share; physical activity; air quality; wildfire smoke; confounding;
fixed effects; calibration; decision curve analysis.

## 1. Introduction

Physical inactivity is a large, preventable health burden, and climate change degrades the conditions
under which health-preserving movement happens. Air quality has become a central worry: wildfire smoke
now routinely pushes eastern US cities into "unhealthy" territory, and a natural question is how much
that smoke suppresses outdoor activity. The usual evidence is an association between *daily* air
quality and *daily* activity. That association is treacherous, because daily air quality is seasonal:
high-AQI days are disproportionately hot, stagnant summer days, which are also the busiest days for
outdoor activity. A regression of activity on daily pollution can therefore report almost anything,
including a protective effect, depending on what it controls for.

This paper makes air quality the object of study and asks whether its effect survives honest
identification. We use municipal bike-share volume as an observable activity proxy for Washington DC
(2022–2024), construct a suppression label from a weather-free, leak-free climatology, and, crucially,
move from daily to *hourly* air quality so pollution can be identified within a day. Within-day
identification holds season, weather regime, day of week, and every other day-level confounder fixed,
and asks only whether the hours of a given day with worse air see less activity.

We judge the forecast the same way: by calibration, decision-relevant net benefit, how much activity a
constrained policy recovers, and whether that policy ever trades safety for participation. The model is
kept minimal (a calibrated logistic regression against a season-aware baseline) because the
contribution is the evaluation and the identification, not model complexity.

Contributions: (i) a city-hour suppression target from a weather-free, leak-free climatology;
(ii) an *identification ladder* for the air-quality effect (marginal, between-day, within-day), showing
that the effect is mostly confounding and that hourly data is needed to see it; (iii) evidence, from a
feature ablation, that air quality adds no forecasting value beyond weather; (iv) evidence that
calibration, not discrimination, governs usability; and (v) Recovered Active Minutes (RAM) under a
safety-constrained, audited time-shift policy.

## 2. Related work

Bike-share demand tracks weather: across forty cities and ~100M trips, usage rises with temperature to
~27–28 °C then falls, with time of day dominant [4]; heat above ~30 °C and rain depress trips [3, 5].
We use municipal bike-share volume as an observable activity proxy, after the canonical
weather-labelled dataset [6]. Heat reduces physical work capacity and raises morbidity [2]; wildfire
smoke's behavioural effect is threshold-like: in a 2019 bushfire experiment children's activity held
until air quality became hazardous, then dropped sharply [1]. Much of the air-quality–activity
literature relies on daily exposure, which is why the confounding we document is easy to miss. That
seasonal confounding of pollution time-series is a known hazard — air-pollution epidemiology developed
case-crossover and time-series designs precisely to remove it — so our contribution is not the
phenomenon but its application here: bringing within-day fixed-effects identification to a mobility
outcome, and pairing it with a calibrated, decision-audited forecast. The evaluation borrows from
clinical prediction, where calibration is a distinct, often-neglected requirement [7, 9], and from
decision-curve analysis [8].

## 3. Problem formulation

The unit is a city-hour. Let `E_t` be weather-free expected ridership: a train-fit diurnal/seasonal
shape (median over `season × daytype × hour`, training years only) scaled by a volume level carried
forward from the last training year, so no test-period information enters the label. An hour is
*active* when `E_t` clears a floor; for active hours,

```
Y_t = 1  if  rides_t < rho * E_t      (rho = 0.5)
```

The forecasting target is `P(Y_t = 1 | W_t)`, with features heat index, two temperature hinges
(`max(0, 55−T)`, `max(0, HI−85)`), hourly AQI, humidity, wind, precipitation, visibility, a smoke flag,
and cyclical hour and weekend. For the estimation question we instead model the continuous ride ratio
`rides_t / E_t` on AQI under increasingly strict identification (Section 5.2). Because `E_t` conditions
on season, the label and the within-day estimator both measure within-season deviation; the forecast
coefficients are associational.

## 4. Data

All sources are public and downloaded by the pipeline.

- **Activity** — Capital Bikeshare trips (2022–2024), aggregated to hourly counts by rider type.
- **Weather** — NOAA Local Climatological Data (Reagan National): hourly temperature, humidity, wind,
  visibility, precipitation, and present-weather codes; temperature and humidity give the NWS heat index.
- **Air quality** — *hourly* US AQI and PM2.5 from the Copernicus Atmosphere Monitoring Service (CAMS)
  reanalysis via Open-Meteo, anchored to EPA AirData ground-station daily AQI. The hourly series covers
  ~80% of city-hours (CAMS hourly begins August 2022); EPA daily AQI fills the remainder and serves as
  the authoritative cross-check for the smoke event.

Activity (DST-aware local) and weather (local standard time) are converted to UTC and joined on the
hour; hourly AQI joins on the local hour, daily AQI on the local day. The 2024 trip files switched to
fractional-second timestamps, parsed explicitly so no rides are dropped. Moving from daily to hourly
AQI is the single change that makes within-day identification possible.

## 5. Methods

### 5.1 Forecasting and calibration

The season-aware climatology baseline predicts the training suppression rate per `season × daytype ×
hour` cell (no weather). The model is a standardized logistic regression, fit unweighted and
class-weighted; the served model (used for RAM and the app) is the unweighted one, with a
gradient-boosting comparator to test for residual nonlinearity. We report Brier, log loss, ECE, and
calibration slope/intercept, and apply 5-fold cross-validated isotonic calibration to the
class-weighted model. Training is 2022–2023, test 2024; served-model metrics carry 95%
percentile-bootstrap CIs (1,000 resamples; RAM resampled over days). A feature ablation adds the groups
temporal → weather → air quality in turn and reports the out-of-time change in each metric, isolating
what air quality contributes.

### 5.2 The air-quality identification ladder

We estimate the air-quality effect on the ride ratio at three levels of rigor:

1. **Marginal** — the raw suppression-rate-by-AQI curve, with no controls.
2. **Between-day (controlled)** — a daily regression of the ride ratio on daily-peak AQI, controlling
   for temperature, precipitation, wind, humidity, weekend, and season, across all 1,096 days. This is
   the typical specification in the literature.
3. **Within-day (fixed effects)** — an hourly regression of the ride ratio on hourly AQI with day and
   hour-of-day fixed effects and hourly weather controls. Day fixed effects absorb season, weather
   regime, day of week, and every other day-level confounder; the AQI coefficient is identified purely
   from intraday variation. CIs use a day-clustered bootstrap.

As a fourth check we match high-AQI hours (AQI ≥ 100) to clean hours of the same season and hour and
compare ride ratios directly.

### 5.3 Recovered activity and safety

The recommender keeps each active hour or shifts it to the lowest-risk hour within ±3 h under a hard
safety envelope (heat index < 103 °F, AQI < 150) and a minimum-benefit rule; RAM = `Σ E_t·(risk_keep −
risk_chosen)`, an upper bound that assumes full, uncapped demand transfer. We audit that no
recommendation falls in unsafe conditions, report operating points at explicit cost ratios, stratify by
season and rider type, and refit the whole pipeline on Seoul.

## 6. Results

The panel holds 26,288 city-hours; 24,354 clear the activity floor, of which 1,466 (6.0%) are
suppressed. We train on 2022–2023 (16,150 hours) and test on 2024 (8,204 hours, 2.5% suppressed).

### 6.1 Discrimination and calibration

| Model | AUROC | AUPRC | Brier | ECE | Cal. slope / int. |
| --- | --- | --- | --- | --- | --- |
| Season-aware climatology | 0.692 | 0.05 | 0.027 | 0.053 | 1.04 / −1.14 |
| Logistic (unweighted, served) | 0.936 | 0.49 | 0.021 | 0.044 | 1.06 / −1.45 |
| Logistic (class-weighted) | 0.940 | 0.45 | 0.128 | 0.253 | 0.69 / −3.83 |
| Logistic (class-weighted) + calibration | 0.940 | 0.47 | 0.029 | 0.064 | 1.21 / −1.78 |
| Gradient boosting | 0.941 | 0.53 | 0.025 | 0.046 | 1.02 / −1.71 |

*Test year 2024, n = 8,204. Served-model 95% CIs (day-clustered bootstrap): AUROC 0.90–0.97, AUPRC
0.32–0.64, Brier 0.017–0.026, ECE 0.036–0.052.*

Features lift AUROC from 0.69 to 0.94; gradient boosting ties the logistic (0.94 each, edging only
AUPRC). But discrimination is not the point. Class weighting, a common default for imbalance, leaves
the model badly over-confident (Brier 0.128, worse than the baseline's 0.027; ECE 0.253), while the
unweighted served model is well calibrated (Brier 0.021, ECE 0.044), and post-hoc isotonic calibration
also repairs the weighted model (ECE 0.253 → 0.064; Figure `reliability`). Net benefit is positive
across low-to-moderate thresholds and maximized at the lowest ones (Figure `decision_curve`); at a 10:1
cost ratio (missed suppression vs unnecessary shift) the threshold is 0.09, flagging 20% of hours at
0.90 sensitivity and 0.82 specificity. Discrimination across all models is shown in Figure `roc`.

### 6.2 Air quality adds no forecasting value

| Feature set | AUROC | AUPRC | Brier | ECE |
| --- | --- | --- | --- | --- |
| Temporal only | 0.509 | 0.03 | 0.027 | 0.053 |
| + Weather | 0.936 | 0.49 | 0.021 | 0.044 |
| + Air quality | 0.936 | 0.49 | 0.021 | 0.044 |

*Out-of-time, 2024.* Weather carries the entire forecast; adding hourly AQI and the smoke flag changes
AUROC and AUPRC by ≈ 0. Refitting the served model with the daily measure swapped in for hourly AQI
flips its standardized coefficient from +0.04 (hourly, correctly signed) to −0.04 (daily) — the daily
measure does not even get the sign right (Table `aqi_coefficient`). The forecast's accuracy is real but
it is a weather forecast.

### 6.3 The air-quality effect is mostly confounding

| Identification | AQI effect (ride-ratio pts per +50 AQI) | 95% CI |
| --- | --- | --- |
| Marginal (no controls) | positive — apparently *protective* | (seasonal confound) |
| Between-day (controlled) | −10.9 | −14.6 to −7.3 |
| Within-day (fixed effects) | −2.4 | −5.9 to +1.1 |

The marginal curve falls as AQI rises (Figure `exposure_response`): suppression drops from ~7.5% in
clean air to near zero above 150 AQI, because dirty hours are summer hours with peak ridership — taken
at face value, pollution looks *good* for activity. Controlling for weather and season across 1,096
days flips the sign to a large negative effect (−10.9 points per +50 AQI). But that estimate is still
identified from between-day variation, and when day fixed effects absorb every day-level confounder —
on the 881 days that carry genuine intraday AQI variation — the effect collapses to −2.4 points with a
CI that crosses zero (Figure `aqi_identification`). This is not lost power: the within-day SE (1.8
points) is no larger than the between-day SE (1.9), and the design's 80%-power threshold (~5 points)
sits well below the −10.9-point between-day estimate, so a between-day-sized effect would have shown up
intraday — its disappearance reflects de-confounding, not insufficient data. Matching high-AQI hours
(AQI ≥ 100; 669 hours over 101 days, median
AQI 118) to clean hours of the same season and hour shows no reduction (ride ratio 1.01× of clean, 95%
CI 0.94–1.09, day-clustered), and the null holds at AQI thresholds of 80, 100, and 120. The honest
reading is that pollution's effect on this mobility proxy is, at most, small, and that the large effects
a daily analysis reports are an artifact of season.

The June 2023 Canadian-wildfire smoke is the vivid exception that proves the rule: on 8 June, daily
ridership fell to 0.76× of expected. But as a single day it is only the 6th-lowest of 66 summer
weekdays and coincided with official advisories, so it cannot carry a population claim; the within-day
estimate is the credible evidence, and it is modest (Figure `smoke_event`).

### 6.4 What does suppress activity

With precipitation and a cold-stress hinge the served model recovers correctly-signed weather drivers:
high humidity (+0.96), cold stress (+0.54), and rain (+0.40) raise suppression, while heat loads
negative (heat index −0.69, heat hinge −0.81) because hot hours coincide with peak summer ridership;
all coefficients are associational. Cold and rain, not heat or smoke, dominate suppression in this city.

### 6.5 Recovered activity and safety

The policy could recover ~37% (95% CI 33–41%) of otherwise-lost activity — about 93,000 rides
(~1.2M rider-minutes) against ~251,000 lost under no adaptation — an upper bound under perfect,
uncapped demand transfer (1,111 shifts into 755 distinct slots). It is conservative (14% of hours
shifted, 0.06% cancelled) and safe by construction: no recommendation falls in unsafe conditions, and
each shift lowers predicted risk by 0.16 on average (Figure `ram_by_month`).

### 6.6 Robustness, transfer, and subgroups

Results are stable to the analyst choices, which we sweep rather than fix silently. Discrimination holds
across the label threshold (ρ ∈ {0.4, 0.5, 0.6}: base rate 3.8–8.9%, AUROC 0.92–0.95, ECE ≤ 0.07), the
activity floor (10/20/30 expected rides: AUROC 0.93–0.94), and the seasons (AUROC 0.90–0.96). The
air-quality null is stable to the high-AQI cutoff (thresholds 80/100/120 all give ride ratio ≈ 1, CIs
spanning 1). RAM is the one number that moves with its tuning: 30%, 37%, 42% for a ±2/±3/±4 h shift
window — monotone and safe at every setting, but a reminder that it is a policy upper bound, not a point
estimate. Refitting on the Seoul Bike dataset and evaluating on a random 25% hold-out gives AUROC 0.87
(ECE 0.02, n = 2,117) — a cross-city method check, since a one-year panel cannot support a clean
out-of-time tail. Burden is unequal: casual riders are suppressed 9.5% of active hours versus 5.1% for
members (~1.9×).

## 7. Discussion

Two results matter beyond this city. First, the air-quality effect on mobility is largely a measurement
artifact: the same data yields a *protective* marginal association, a large negative between-day effect,
and a bounded-near-null within-day effect, and only the last holds season fixed. Studies that regress
activity on daily pollution (the norm) are therefore likely to overstate the effect, and hourly data is
what exposes the gap. Second, calibration, not discrimination, is what makes a forecast usable: a model
excellent on AUROC is unusable when class-weighted, and reporting discrimination alone would have hidden
a Brier worse than the baseline. Despite a deliberately modest model, a calibrated risk score plus a
safety-constrained policy recovers a meaningful, bounded share of lost activity and transfers to a
second city. The PulseShift app ships exactly this served model client-side; this paper is its evidence
layer.

## 8. Limitations

Construct validity is the central caveat: bike-share ridership is a proxy for *outdoor mobility demand*,
not measured exercise, and it bundles commuting, errands, and tourism with recreation. The suppression
label is a constructed cutoff (ridership below half of climatology), not an observed skipped session. We
therefore read every result as a statement about weather- and air-quality-driven mobility, and any
physical-activity or health interpretation requires external validation against measured activity (e.g.
wearable or survey data) — which a multi-city replication should carry. Hourly AQI is CAMS reanalysis,
which underestimates localized smoke plumes relative to ground stations (it reads the 8 June peak near
150 AQI where the EPA station recorded 196), so the within-day estimate is conservative for exactly the
events of interest; ground-station hourly data would sharpen it. The within-day estimator is
fit on the 881 of 1,096 days that carry genuine intraday AQI variation — days before CAMS hourly
coverage begins (August 2022) fall back to a flat daily value and are excluded, since they add no
identifying within-day variation. Carrying the 2024 volume forward to stay leak-free makes the 2024
suppression rate a modest undercount. RAM is a model-based upper bound. Equity is behavioural (rider
type), not demographic. The smoke episode is a single confounded event; the within-day regression
carries that claim.

## 9. Reproducibility

Public data and pinned dependencies. `run_analysis.py` regenerates every figure and table from the
committed panel, `build_data.py` rebuilds that panel from source, `train_model.py` exports the served
model, and `validate_seoul.py` runs the external check; `make all` runs setup, analysis, model export,
and tests (the committed panel makes the data step optional). The primary specification, and the line
between confirmatory and exploratory results, are declared in `paper/preregistration.md`.
`pytest` (in CI) guards the heat index, the leak-free label, the within-day estimator, the safety
policy, and model-export parity. See `research/README.md`.

## References

1. Del Pozo Cruz B, Hartwig TB, Sanders T, et al. The effects of the Australian bushfires on physical
   activity in children. *Environment International*. 2020;146:106214. doi:10.1016/j.envint.2020.106214
2. Ebi KL, Capon A, Berry P, et al. Hot weather and heat extremes: health risks. *The Lancet*.
   2021;398(10301):698–708. doi:10.1016/S0140-6736(21)01208-3
3. Kim K. Investigation on the effects of weather and calendar events on bike-sharing according to the
   trip patterns of bike rentals of stations. *Journal of Transport Geography*. 2018.
4. Bean R, et al. How does weather affect bikeshare use? A comparative analysis of forty cities across
   climate zones. *Journal of Transport Geography*. 2021.
5. Guzel D, et al. Assessment of weather-driven travel behavior on a small-scale docked bike-sharing
   system usage. *Travel Behaviour and Society*. 2025.
6. Fanaee-T H, Gama J. Event labeling combining ensemble detectors and background knowledge.
   *Progress in Artificial Intelligence*. 2013. doi:10.1007/s13748-013-0040-3
7. Staffa SJ, Zurakowski D. Statistical development and validation of clinical prediction models.
   *Anesthesiology*. 2021;135(3):396–405. doi:10.1097/ALN.0000000000003871
8. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models.
   *Medical Decision Making*. 2006;26(6):565–574. doi:10.1177/0272989X06295361
9. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. *Proceedings
   of the 22nd International Conference on Machine Learning (ICML)*. 2005.
```
