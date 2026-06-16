"""Write a dataframe as paired CSV and markdown tables."""

from . import config


def _fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def write_table(df, name, tables_dir=None):
    tables_dir = tables_dir or config.TABLES
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    (tables_dir / f"{name}.md").write_text("\n".join([header, sep, *rows]) + "\n")
