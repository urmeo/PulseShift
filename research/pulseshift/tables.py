"""Write research tables as CSV."""

from __future__ import annotations

import pandas as pd

from . import config


def write_table(df: pd.DataFrame, name: str, tables_dir=None) -> None:
    tables_dir = tables_dir or config.TABLES
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv", index=False)
