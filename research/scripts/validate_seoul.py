"""External validation on a second city: Seoul Bike Sharing (UCI #560)."""

import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pulseshift import config
from pulseshift.evaluation import metrics
from pulseshift.features import heat_index_f
from pulseshift.panel import suppression_mask
from pulseshift.tables import write_table

URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
FEATURES = [
    "heat_index_f",
    "cold_stress",
    "heat_stress",
    "humidity",
    "wind_mph",
    "precip_in",
    "visibility_mi",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]


def _load():
    path = config.RAW / "seoul.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            URL, headers={"User-Agent": "pulseshift-research/1.0"}
        )
        path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV in Seoul archive from {URL}")
        return pd.read_csv(z.open(names[0]), encoding="latin-1")


def main():
    df = _load()
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        for key, std in [
            ("rented", "rides"),
            ("temperature", "temp_c"),
            ("humidity", "humidity"),
            ("wind", "wind_ms"),
            ("visibility", "visibility"),
            ("rainfall", "rain_mm"),
            ("seasons", "season"),
            ("functioning", "functioning"),
        ]:
            if cl.startswith(key):
                rename[c] = std
        if cl == "hour":
            rename[c] = "hour"
        if cl == "date":
            rename[c] = "date"
    df = df.rename(columns=rename)
    df = df[df["functioning"] == "Yes"].copy()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["daytype"] = np.where(df["is_weekend"] == 1, "weekend", "weekday")
    df["temp_f"] = df["temp_c"] * 9 / 5 + 32
    df["heat_index_f"] = heat_index_f(df["temp_f"], df["humidity"])
    df["cold_stress"] = (config.COLD_STRESS_BASE_F - df["temp_f"]).clip(lower=0)
    df["heat_stress"] = (df["heat_index_f"] - config.HEAT_STRESS_BASE_F).clip(lower=0)
    df["wind_mph"] = df["wind_ms"] * 2.237
    df["precip_in"] = df["rain_mm"] / 25.4
    df["visibility_mi"] = df["visibility"] * 10 / 1609
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # cross-city check of the method: a random 75/25 hold-out (a chronological tail on one
    # year would put an entire unseen season in the test set). Split once, before climatology.
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    df["is_train"] = df.index < int(len(df) * 0.75)
    shape = df[df["is_train"]].groupby(["season", "daytype", "hour"])["rides"].median()
    df["expected"] = shape.reindex(
        pd.MultiIndex.from_frame(df[["season", "daytype", "hour"]])
    ).to_numpy()
    df = df.dropna(subset=["expected"])
    df = df[df["expected"] >= config.EXPECTED_FLOOR].reset_index(drop=True)
    _, suppressed = suppression_mask(df["rides"].to_numpy(), df["expected"].to_numpy())
    df["suppressed"] = suppressed.astype(int)

    tr, te = df[df["is_train"]], df[~df["is_train"]]
    model = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    model.fit(tr[FEATURES], tr["suppressed"])
    p = model.predict_proba(te[FEATURES])[:, 1]
    m = metrics(te["suppressed"], p)

    row = pd.DataFrame(
        [
            {
                "city": "Seoul",
                "n_test": m["n"],
                "base_rate": round(m["base_rate"], 3),
                "auroc": round(m["auroc"], 3),
                "brier": round(m["brier"], 3),
                "ece": round(m["ece"], 3),
            }
        ]
    )
    write_table(row, "seoul_validation")
    print(row.to_string(index=False))


if __name__ == "__main__":
    main()
