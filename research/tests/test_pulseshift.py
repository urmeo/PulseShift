"""Regression tests for the PulseShift pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from pulseshift import config, ram, safety
from pulseshift.evaluation import bootstrap_ci, metrics
from pulseshift.features import MODEL_FEATURES, heat_index_f
from pulseshift.models import fit_logistic
from pulseshift.panel import _expected_rides, active, load_panel


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
    exp = _expected_rides(base)
    # shape = median(100,200)=150; 2024 carries 2023 level (200/150) -> 200
    assert exp[2] == pytest.approx(200.0, rel=1e-6)
    # changing the test year's volume must not change its expectation
    moved = base.copy()
    moved.loc[2, "rides_total"] = 99999
    assert _expected_rides(moved)[2] == pytest.approx(exp[2], rel=1e-6)


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


def test_served_model_matches_export():
    """model.json must reproduce a fresh unweighted fit."""
    model_path = config.ROOT.parent / "model.json"
    if not model_path.exists():
        pytest.skip("model.json not built")
    m = json.loads(model_path.read_text())
    assert list(m["features"]) == MODEL_FEATURES
    clf = fit_logistic(active(load_panel()), balanced=False).named_steps["clf"]
    assert np.allclose(clf.coef_[0], m["coef"], atol=1e-6)
