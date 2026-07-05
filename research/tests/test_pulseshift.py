"""Regression tests for the PulseShift pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from pulseshift import airquality, config, ram, safety
from pulseshift.evaluation import bootstrap_ci, metrics
from pulseshift.features import MODEL_FEATURES, heat_index_f
from pulseshift.models import fit_logistic
from pulseshift.panel import active, expected_rides, load_panel


def test_heat_index_identity_and_amplifies():
    assert float(heat_index_f(75, 30)) == 75
    assert float(heat_index_f(95, 70)) > 95


def test_label_climatology_is_leak_free():
    """Test-year expectation must not use test-year data."""
    base = pd.DataFrame(
        {
            "year": [2022, 2023, 2024],
            "season": ["summer"] * 3,
            "daytype": ["weekday"] * 3,
            "hour": [8] * 3,
            "rides_total": [100, 200, 1000],
        }
    )
    exp = expected_rides(base)
    # shape = median(100,200)=150; 2024 carries 2023 level (200/150) -> 200
    assert exp[2] == pytest.approx(200.0, rel=1e-6)
    # changing the test year's volume must not change its expectation
    moved = base.copy()
    moved.loc[2, "rides_total"] = 99999
    assert expected_rides(moved)[2] == pytest.approx(exp[2], rel=1e-6)


def test_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))["auroc"] == 0.5


def test_bootstrap_ci_ordered():
    y = np.array([0, 0, 1, 1])
    from sklearn.metrics import roc_auc_score

    lo, hi = bootstrap_ci(
        y, np.array([0.1, 0.2, 0.8, 0.9]), roc_auc_score, n=50, require_two_classes=True
    )
    assert lo <= hi


def test_policy_never_recommends_unsafe_hour():
    df = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                ["2024-07-01 08:00", "2024-07-01 09:00", "2024-07-01 10:00"]
            ),
            "hour": [8, 9, 10],
            "risk": [0.5, 0.1, 0.6],
            "heat_index_f": [90, 88, 95],
            "aqi": [60, 60, 60],
            "expected_rides": [100, 100, 100],
        }
    )
    audit = safety.audit(ram.recommend(df))
    assert audit["unsafe_recommendations"] == 0 and audit["all_safe"]


def test_recommend_is_row_order_invariant():
    """Results must attach to the right rows regardless of input ordering."""
    base = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                [
                    "2024-07-01 08:00",
                    "2024-07-01 09:00",
                    "2024-07-02 08:00",
                    "2024-07-02 09:00",
                ]
            ),
            "hour": [8, 9, 8, 9],
            "risk": [0.5, 0.1, 0.6, 0.2],
            "heat_index_f": [90, 88, 92, 89],
            "aqi": [60, 60, 60, 60],
            "expected_rides": [100, 100, 100, 100],
        }
    )
    order = [2, 0, 3, 1]  # interleave the two days
    shuffled = base.iloc[order].reset_index(drop=True)

    sorted_reco = ram.recommend(base).set_index("ts_local")
    shuffled_reco = ram.recommend(shuffled).set_index("ts_local")

    cols = ["action", "target_hour", "chosen_risk"]
    pd.testing.assert_frame_equal(
        sorted_reco[cols].sort_index(), shuffled_reco[cols].sort_index()
    )
    # a keep row must target its own hour (internal consistency)
    for reco in (sorted_reco, shuffled_reco):
        keep = reco[reco["action"] == "keep"]
        assert (keep["target_hour"] == keep["hour"]).all()


def test_hourly_aqi_join():
    """Panel AQI uses the hourly series where present, daily fallback otherwise."""
    panel = load_panel()
    assert "aqi_hourly" in panel.columns
    present = panel["aqi_hourly"].notna()
    assert present.mean() > 0.5
    assert (panel.loc[present, "aqi"] == panel.loc[present, "aqi_hourly"]).all()


def _synthetic_aqi_panel():
    """Within-day AQI effect plus a day-level confounder correlated with daily AQI."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(12):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        day_mean = 20 * d  # rising daily AQI
        day_boost = 0.02 * d  # confounder: busy days are also dirty days
        for h in range(24):
            aqi = day_mean + 40 + 20 * np.sin(2 * np.pi * h / 24)
            ratio = 1.0 - 0.004 * aqi + day_boost + rng.normal(0, 0.01)
            rows.append(
                {
                    "ts_local": start + pd.Timedelta(hours=h),
                    "hour": h,
                    "aqi": aqi,
                    "rides_total": 100 * ratio,
                    "expected_rides": 100.0,
                    "temp_f": rng.normal(50, 2),
                    "precip_in": 0.0,
                    "wind_mph": rng.normal(8, 1),
                    "humidity": rng.normal(60, 3),
                    "season": "winter",
                    "is_weekend": int(start.dayofweek >= 5),
                }
            )
    return pd.DataFrame(rows)


def test_within_day_removes_day_confounder():
    """Within-day FE recovers the true -0.2/+50 slope; between-day is biased up."""
    panel = _synthetic_aqi_panel()
    within = airquality.within_day_effect(panel, n_boot=200)
    between = airquality.between_day_effect(panel, n_boot=200)
    assert within["effect_per_50"] == pytest.approx(-0.2, abs=0.05)
    assert within["ci_low"] <= within["effect_per_50"] <= within["ci_high"]
    assert between["effect_per_50"] > within["effect_per_50"]
    assert within["mde80"] > within["se"] > 0  # detectable effect scales with SE


def test_smoke_episode_ci_ordered():
    e = airquality.smoke_episodes(_synthetic_aqi_panel(), aqi_thresh=100, n_boot=200)
    assert e["polluted_hours"] > 0
    assert e["ci_low"] <= e["ride_ratio_vs_clean"] <= e["ci_high"]


def test_measurement_error_deattenuates():
    """Lower reliability scales the corrected effect further from zero."""
    panel = load_panel()
    panel["aqi_hourly"] = panel["aqi"]  # synthetic perfect hourly coverage
    panel["aqi_epa_daily"] = panel["aqi"]
    out = airquality.measurement_error_bound(panel, beta_per_50=-2.0)
    by_rho = {
        r["rho"]: r["corrected_per_50"]
        for r in out["rows"]
        if r["reliability"] == "assumed"
    }
    assert by_rho[0.5] < by_rho[0.7] < 0  # more error -> larger magnitude


def test_served_model_matches_export():
    """model.json must reproduce a fresh unweighted fit: features, scaler, and coefficients.

    The browser computes (x - mean) / scale then a dot product with coef plus
    intercept, so every one of those arrays must match what the app ships, not
    only the coefficients.
    """
    model_path = config.ROOT.parent / "model.json"
    if not model_path.exists():
        pytest.skip("model.json not built")
    m = json.loads(model_path.read_text())
    assert list(m["features"]) == MODEL_FEATURES
    model = fit_logistic(active(load_panel()), balanced=False)
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]
    assert np.allclose(scaler.mean_, m["mean"], atol=1e-6)
    assert np.allclose(scaler.scale_, m["scale"], atol=1e-6)
    assert np.allclose(clf.coef_[0], m["coef"], atol=1e-6)
    assert clf.intercept_[0] == pytest.approx(m["intercept"], abs=1e-6)


def test_export_constants_match_config():
    """Exported safety + thermal-stress hinges must match config so the app can't drift."""
    model_path = config.ROOT.parent / "model.json"
    if not model_path.exists():
        pytest.skip("model.json not built")
    m = json.loads(model_path.read_text())
    assert m["safety"]["heat_unsafe_f"] == config.HEAT_UNSAFE_F
    assert m["safety"]["aqi_unsafe"] == config.AQI_UNSAFE
    assert m["stress"]["cold_base_f"] == config.COLD_STRESS_BASE_F
    assert m["stress"]["heat_base_f"] == config.HEAT_STRESS_BASE_F
