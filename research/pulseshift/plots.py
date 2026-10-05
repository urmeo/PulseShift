"""Publication figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .calibration import reliability
from .decision import net_benefit

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


def reliability_plot(y, balanced, served) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    for prob, lab, color in [
        (balanced, "Balanced", "#c44"),
        (served, "Unweighted (holdout)", "#2a7"),
    ]:
        mp, fp = reliability(y, prob, n_bins=10)
        ax.plot(mp, fp, "o-", color=color, label=lab)
    ax.set_xlabel("Predicted suppression probability")
    ax.set_ylabel("Observed suppression rate")
    ax.set_title("2024 holdout calibration")
    ax.legend()
    _save(fig, "reliability.png")


def roc_plot(curves: dict) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    for label, (y, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUROC {auc:.2f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discrimination")
    ax.legend()
    _save(fig, "roc.png")


def decision_plot(y, p) -> None:
    thresholds = np.linspace(0.01, 0.6, 60)
    model, treat_all = net_benefit(y, p, thresholds)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(thresholds, model, color="#2a7", label="Model")
    ax.plot(thresholds, treat_all, color="#888", lw=1, label="Flag all")
    ax.axhline(0, color="#333", lw=1, label="Flag none")
    ax.set_ylim(min(0, model.min()) - 0.01, model.max() + 0.02)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve")
    ax.legend()
    _save(fig, "decision_curve.png")


def exposure_response(panel: pd.DataFrame) -> None:
    df = panel.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, label, edges in [
        (axes[0], "heat_index_f", "Heat index (F)", np.arange(40, 115, 5)),
        (axes[1], "aqi", "Air Quality Index", np.arange(0, 320, 20)),
    ]:
        binned = pd.cut(df[col], edges)
        rate = df.groupby(binned, observed=True)["suppressed"].mean()
        centers = [iv.mid for iv in rate.index]
        ax.plot(centers, rate.values, "o-", color="#36c")
        ax.set_xlabel(label)
        ax.set_ylabel("Suppression rate")
    axes[0].set_title("Heat response")
    axes[1].set_title("Air-quality association (unadjusted)")
    _save(fig, "exposure_response.png")


def aqi_identification_plot(between: dict, within: dict) -> None:
    """Daily-peak and intraday AQI associations with 95% bootstrap intervals."""
    rows = [
        ("Between-day\n(daily peak)", between),
        ("Within-day\n(hourly)", within),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    for y, (label, est) in enumerate(rows):
        ax.errorbar(
            est["effect_per_50"],
            y,
            xerr=[
                [est["effect_per_50"] - est["ci_low"]],
                [est["ci_high"] - est["effect_per_50"]],
            ],
            fmt="o",
            color="#36c",
            capsize=4,
        )
        ax.text(est["effect_per_50"], y + 0.18, label, ha="center", fontsize=9)
    ax.axvline(0, color="#a33", ls="--", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("Ride-ratio change per +50 AQI")
    ax.set_title("AQI associations: different exposures and controls")
    _save(fig, "aqi_identification.png")


def smoke_event(
    panel: pd.DataFrame, start: str = "2023-06-05", end: str = "2023-06-12"
) -> None:
    df = panel[(panel["ts_local"] >= start) & (panel["ts_local"] < end)]
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df["ts_local"], df["aqi"], color="#a33", label="AQI")
    ax1.set_ylabel("Air Quality Index", color="#a33")
    ax1.axhline(config.AQI_UNSAFE, ls="--", color="#a33", lw=1)
    ax2 = ax1.twinx()
    ax2.plot(df["ts_local"], df["rides_total"], color="#36c", alpha=0.8, label="Rides")
    ax2.set_ylabel("Hourly rides", color="#36c")
    ax1.set_title("June 2023 wildfire smoke, Washington DC")
    fig.autofmt_xdate()
    _save(fig, "smoke_event.png")


def ram_by_month(reco: pd.DataFrame, recovered: pd.Series) -> None:
    df = reco.assign(recovered=recovered.values)
    df["month"] = df["ts_local"].dt.strftime("%Y-%m")
    monthly = df.groupby("month")["recovered"].sum()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(monthly.index, monthly.values, color="#2a7")
    ax.set_ylabel("Ride-weighted predicted-risk reduction")
    ax.set_title("Model-based time-shift scenario by month")
    fig.autofmt_xdate()
    _save(fig, "ram_by_month.png")
