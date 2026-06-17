"""Train, evaluate, and produce every paper figure and table.

Orchestrated as a sequence of small steps so each analysis block is readable on its own:
fit -> compare -> calibrate CIs -> coefficients -> decision/cost -> figures -> RAM ->
smoke event -> AQI identification -> AQI coefficient -> ablation -> subgroups ->
threshold sensitivity -> summary.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from pulseshift import airquality, config, decision, equity, plots, ram, safety
from pulseshift.calibration import calibrate_cv
from pulseshift.evaluation import bootstrap_ci, expected_calibration_error, metrics
from pulseshift.features import MODEL_FEATURES
from pulseshift.models import (
    ClimatologyBaseline,
    fit_gbm,
    fit_logistic,
    predict,
    temporal_split,
)
from pulseshift.panel import active, label_suppression, load_panel
from pulseshift.tables import write_table as _write_table


def fit_models(train_all, work, test):
    """Fit baselines + models; return test-set predictions and the served model."""
    clim = ClimatologyBaseline().fit(train_all)
    logit_unw = fit_logistic(train_all, balanced=False)
    logit = fit_logistic(train_all, balanced=True)
    calibrated = calibrate_cv(train_all, method="isotonic", cv=5, balanced=True)
    served = fit_logistic(work, balanced=False)  # all-data model shipped in the app
    preds = {
        "clim": clim.predict_proba(test),
        "unw": predict(logit_unw, test),
        "bal": predict(logit, test),
        "cal": calibrated.predict_proba(test[MODEL_FEATURES])[:, 1],
        "gbm": predict(fit_gbm(train_all), test),
    }
    return preds, served


def model_comparison(test, preds):
    comparison = pd.DataFrame(
        [
            {"model": "Climatology", **metrics(test["suppressed"], preds["clim"])},
            {
                "model": "Logistic (unweighted)",
                **metrics(test["suppressed"], preds["unw"]),
            },
            {
                "model": "Logistic (balanced)",
                **metrics(test["suppressed"], preds["bal"]),
            },
            {
                "model": "Logistic (balanced) + calibration",
                **metrics(test["suppressed"], preds["cal"]),
            },
            {"model": "Gradient boosting", **metrics(test["suppressed"], preds["gbm"])},
        ]
    )
    _write_table(comparison, "model_comparison")
    return comparison


def served_confidence_intervals(test, p_unw):
    yt = test["suppressed"].to_numpy()
    days = test["ts_local"].dt.normalize().to_numpy()  # cluster by day
    ci = {
        "auroc": bootstrap_ci(
            yt, p_unw, roc_auc_score, require_two_classes=True, groups=days
        ),
        "auprc": bootstrap_ci(
            yt, p_unw, average_precision_score, require_two_classes=True, groups=days
        ),
        "brier": bootstrap_ci(yt, p_unw, brier_score_loss, groups=days),
        "ece": bootstrap_ci(
            yt,
            p_unw,
            lambda y, p: expected_calibration_error(np.asarray(y), np.asarray(p)),
            groups=days,
        ),
    }
    _write_table(
        pd.DataFrame(
            [{"metric": k, "low": lo, "high": hi} for k, (lo, hi) in ci.items()]
        ),
        "served_ci",
    )
    return ci


def coefficients_and_exposure(work, served):
    coef = pd.DataFrame(
        {"feature": MODEL_FEATURES, "coefficient": served.named_steps["clf"].coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)
    _write_table(coef, "logistic_coefficients")

    rows = []
    for var, edges in [
        ("heat_index_f", [0, 40, 55, 70, 80, 90, 130]),
        ("aqi", [0, 50, 75, 100, 150, 400]),
    ]:
        for interval, g in work.groupby(pd.cut(work[var], edges), observed=True):
            rows.append(
                {
                    "variable": var,
                    "bin": str(interval),
                    "n": len(g),
                    "suppression_rate": g["suppressed"].mean(),
                }
            )
    _write_table(pd.DataFrame(rows), "exposure_response")


def decision_and_cost(test, p_unw):
    """Decision-curve net benefit, plus operating points at explicit cost ratios."""
    yt = test["suppressed"].to_numpy()
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    nb_model, nb_all = decision.net_benefit(test["suppressed"], p_unw, thresholds)
    _write_table(
        pd.DataFrame({"threshold": thresholds, "model": nb_model, "adapt_all": nb_all}),
        "decision_curve",
    )

    # net benefit is maximized at the lowest thresholds, so pick thresholds from a stated
    # cost ratio (missed suppression vs unnecessary shift): threshold = 1 / (1 + ratio).
    cost_rows = []
    for ratio in (5, 10, 20):
        t = 1 / (1 + ratio)
        flag = p_unw >= t
        tn, fp, fn, tp = confusion_matrix(yt, flag).ravel()
        cost_rows.append(
            {
                "cost_ratio_miss_to_flag": ratio,
                "threshold": round(t, 3),
                "sensitivity": round(tp / (tp + fn), 3),
                "specificity": round(tn / (tn + fp), 3),
                "flagged_share": round(float(flag.mean()), 3),
            }
        )
    _write_table(pd.DataFrame(cost_rows), "cost_threshold")
    return cost_rows


def figures(test, panel, work, preds, comparison):
    yt = test["suppressed"].to_numpy()
    plots.reliability_plot(yt, preds["bal"], preds["unw"])
    plots.roc_plot(
        {
            "Climatology": (
                test["suppressed"],
                preds["clim"],
                comparison.iloc[0]["auroc"],
            ),
            "Balanced": (test["suppressed"], preds["bal"], comparison.iloc[2]["auroc"]),
            "Unweighted (served)": (
                test["suppressed"],
                preds["unw"],
                comparison.iloc[1]["auroc"],
            ),
        }
    )
    plots.decision_plot(yt, preds["unw"])
    plots.exposure_response(work)
    plots.smoke_event(panel)


def recovered_active_minutes(test, p_unw):
    """Time-shift policy + safety audit; RAM% bootstrapped over days."""
    test = test.assign(risk=p_unw)
    reco = ram.recommend(test)
    ram_stats = ram.ram_table(reco)
    plots.ram_by_month(reco, ram_stats["per_hour"])
    audit = safety.audit(reco)

    by_day = (
        pd.DataFrame(
            {
                "day": reco["ts_local"].dt.date.to_numpy(),
                "recovered": ram_stats["per_hour"].to_numpy(),
                "lost": (reco["expected_rides"] * reco["risk"]).to_numpy(),
            }
        )
        .groupby("day")[["recovered", "lost"]]
        .sum()
    )
    rng = np.random.default_rng(0)
    days = by_day.index.to_numpy()
    ratios = []
    for _ in range(1000):
        g = by_day.loc[rng.choice(days, size=len(days), replace=True)]
        ratios.append(
            g["recovered"].sum() / g["lost"].sum() if g["lost"].sum() else 0.0
        )
    ram_ci = [round(float(x), 3) for x in np.percentile(ratios, [2.5, 97.5])]
    return test, ram_stats, audit, ram_ci


def smoke_event(panel):
    event = panel[
        (panel["ts_local"] >= "2023-06-06") & (panel["ts_local"] < "2023-06-10")
    ]
    event_tbl = (
        event.groupby(event["ts_local"].dt.date)
        .agg(
            aqi=("aqi", "max"),
            rides=("rides_total", "sum"),
            expected=("expected_rides", "sum"),
        )
        .reset_index()
        .rename(columns={"ts_local": "date"})
    )
    event_tbl["rides_vs_expected"] = (event_tbl["rides"] / event_tbl["expected"]).round(
        2
    )
    _write_table(event_tbl, "smoke_event")

    summer = panel[
        (panel["ts_local"] >= "2023-06-01")
        & (panel["ts_local"] < "2023-09-01")
        & (panel["daytype"] == "weekday")
    ]
    daily_ratio = summer.groupby(summer["ts_local"].dt.date).apply(
        lambda d: d["rides_total"].sum() / d["expected_rides"].sum()
    )
    jun8 = float(daily_ratio.loc[pd.Timestamp("2023-06-08").date()])
    return {
        "jun8_ratio": jun8,
        "summer_weekday_days": int(len(daily_ratio)),
        "pct_days_below_jun8": float((daily_ratio <= jun8).mean()),
        "summer_ratio_median": float(daily_ratio.median()),
    }


def aqi_identification(work):
    """Identification ladder: marginal -> between-day -> within-day fixed effects."""
    between = airquality.between_day_effect(work)
    within = airquality.within_day_effect(work)
    episodes = airquality.smoke_episodes(work, aqi_thresh=100)
    plots.aqi_identification_plot(between, within)

    cols = ["identification", "effect_per_50", "ci_low", "ci_high", "se", "n"]
    ladder = pd.DataFrame(
        [
            {"identification": "between-day (controlled)", **between},
            {"identification": "within-day fixed effects", **within},
        ]
    )[cols]
    _write_table(ladder, "aqi_identification")
    _write_table(pd.DataFrame([episodes]), "smoke_episodes")

    power = pd.DataFrame(
        [
            {
                "estimand": "within-day AQI per +50",
                "estimate": within["effect_per_50"],
                "se": within["se"],
                "mde_80pct_power": within["mde80"],
                "mde_90pct_power": within["mde90"],
            }
        ]
    )
    _write_table(power, "aqi_power")

    error = airquality.measurement_error_bound(work, within["effect_per_50"])
    _write_table(pd.DataFrame(error["rows"]), "measurement_error")
    return {
        "between_day": between,
        "within_day": within,
        "episodes": episodes,
        "measurement_error": error,
    }


def aqi_coefficient(work):
    """Served logistic AQI coefficient: hourly measure vs daily measure."""
    ai = MODEL_FEATURES.index("aqi")
    hourly = fit_logistic(work, balanced=False)
    daily = fit_logistic(work.assign(aqi=work["aqi_epa_daily"]), balanced=False)
    rows = [
        {
            "aqi_source": "hourly (served)",
            "std_coefficient": float(hourly.named_steps["clf"].coef_[0][ai]),
        },
        {
            "aqi_source": "daily",
            "std_coefficient": float(daily.named_steps["clf"].coef_[0][ai]),
        },
    ]
    _write_table(pd.DataFrame(rows), "aqi_coefficient")
    return rows


def feature_ablation(train_all, test):
    """Marginal value of each feature group, out-of-time."""
    groups = {
        "temporal": ["hour_sin", "hour_cos", "is_weekend"],
        "+ weather": [
            "heat_index_f",
            "cold_stress",
            "heat_stress",
            "humidity",
            "wind_mph",
            "precip_in",
            "visibility_mi",
        ],
        "+ air quality": ["aqi", "smoke_haze"],
    }
    cols, rows = [], []
    for name, feats in groups.items():
        cols += feats
        m = metrics(
            test["suppressed"],
            predict(
                fit_logistic(train_all, balanced=False, features=cols),
                test,
                features=cols,
            ),
        )
        rows.append(
            {
                "features": name,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "brier": m["brier"],
                "ece": m["ece"],
            }
        )
    _write_table(pd.DataFrame(rows), "ablation")
    return rows


def subgroups(test_with_risk, panel):
    _write_table(equity.strata_metrics(test_with_risk, "season"), "equity_by_season")
    _write_table(equity.strata_metrics(test_with_risk, "daytype"), "equity_by_daytype")
    burden = equity.rider_burden(panel)
    _write_table(burden, "rider_burden")
    return burden


def label_sensitivity(work):
    rows = []
    for ratio in config.SENSITIVITY_RATIOS:
        _, lab = label_suppression(work, ratio=ratio)
        tr, te = temporal_split(work.assign(suppressed=lab))
        p = predict(fit_logistic(tr, balanced=False), te)
        row = {"ratio": ratio, "base_rate": float(lab.mean())}
        row.update(
            {k: metrics(te["suppressed"], p)[k] for k in ["auroc", "brier", "ece"]}
        )
        rows.append(row)
    _write_table(pd.DataFrame(rows), "label_sensitivity")


def policy_sensitivity(test, p_unw):
    """RAM and safety across shift-window choices."""
    rows = []
    for window in (2, 3, 4):
        reco = ram.recommend(test.assign(risk=p_unw), window=window)
        stats = ram.ram_table(reco)
        rows.append(
            {
                "shift_window_h": window,
                "ram_pct_of_lost": round(stats["ram_pct_of_lost"], 3),
                "share_shifted": round(stats["share_shifted"], 3),
                "all_safe": safety.audit(reco)["all_safe"],
            }
        )
    _write_table(pd.DataFrame(rows), "sensitivity_policy")
    return rows


def smoke_sensitivity(work):
    """Episode effect across AQI thresholds."""
    rows = []
    for thresh in (80, 100, 120):
        e = airquality.smoke_episodes(work, aqi_thresh=thresh)
        rows.append(
            {
                k: e[k]
                for k in [
                    "aqi_threshold",
                    "polluted_days",
                    "ride_ratio_vs_clean",
                    "ci_low",
                    "ci_high",
                ]
            }
        )
    _write_table(pd.DataFrame(rows), "sensitivity_smoke")
    return rows


def floor_sensitivity(panel):
    """Discrimination across the activity-floor choice."""
    rows = []
    for floor in (10, 20, 30):
        sub = panel[panel["expected_rides"] >= floor].copy()
        sub["suppressed"] = (
            sub["rides_total"] < config.SUPPRESSION_RATIO * sub["expected_rides"]
        ).astype(int)
        tr, te = temporal_split(sub)
        m = metrics(te["suppressed"], predict(fit_logistic(tr, balanced=False), te))
        rows.append(
            {
                "floor": floor,
                "active_hours": int(len(sub)),
                "base_rate": round(m["base_rate"], 3),
                "auroc": round(m["auroc"], 3),
                "ece": round(m["ece"], 3),
            }
        )
    _write_table(pd.DataFrame(rows), "sensitivity_floor")
    return rows


def main():
    panel = load_panel()
    work = active(panel)
    train_all, test = temporal_split(work)

    preds, served = fit_models(train_all, work, test)
    comparison = model_comparison(test, preds)
    ci = served_confidence_intervals(test, preds["unw"])
    coefficients_and_exposure(work, served)
    cost_rows = decision_and_cost(test, preds["unw"])
    figures(test, panel, work, preds, comparison)
    test, ram_stats, audit, ram_ci = recovered_active_minutes(test, preds["unw"])
    smoke_context = smoke_event(panel)
    aqi = aqi_identification(work)
    aqi_coef = aqi_coefficient(work)
    ablation = feature_ablation(train_all, test)
    burden = subgroups(test, panel)
    label_sensitivity(work)
    policy_sens = policy_sensitivity(test, preds["unw"])
    smoke_sens = smoke_sensitivity(work)
    floor_sens = floor_sensitivity(panel)

    summary = {
        "panel_rows": int(len(panel)),
        "active_hours": int(len(work)),
        "train_rows": int(len(train_all)),
        "test_rows": int(len(test)),
        "test_base_rate": float(test["suppressed"].mean()),
        "model_comparison": comparison.to_dict(orient="records"),
        "served_ci": ci,
        "ram": {k: v for k, v in ram_stats.items() if k != "per_hour"},
        "ram_pct_ci": ram_ci,
        "ram_pct_discounted_50": round(0.5 * ram_stats["ram_pct_of_lost"], 3),
        "safety": audit,
        "smoke_event": smoke_context,
        "aqi_identification": aqi,
        "aqi_coefficient": aqi_coef,
        "ablation": ablation,
        "cost_threshold": cost_rows,
        "rider_burden": burden.to_dict(orient="records"),
        "sensitivity": {
            "policy": policy_sens,
            "smoke": smoke_sens,
            "floor": floor_sens,
        },
    }
    (config.TABLES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("model_comparison:", json.dumps(summary["model_comparison"]))
    print(
        "RAM:",
        summary["ram"],
        "\nSafety:",
        summary["safety"],
        "\nSmoke:",
        summary["smoke_event"],
    )


if __name__ == "__main__":
    main()
