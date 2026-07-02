# Contributing to PulseShift

Thanks for your interest. PulseShift is a small, reproducible research project: a calibrated mobility-suppression forecast plus an evidence layer on air-quality confounding. Contributions that improve clarity, reproducibility, or extend the analysis to new cities are especially welcome.

## Development setup

Everything is driven by the Makefile from the repository root:

```bash
make setup     # create .venv and install pinned dependencies
make test      # run the pytest suite
make analysis  # regenerate every paper table and figure (~2-4 min; prints per-stage progress)
make model     # refit and export the served model to model.json / model.js
make all       # setup + analysis + model + test
```

The processed panel (research/data/processed/panel.csv.gz) is committed, so analysis and tests run without re-downloading raw data. make data rebuilds the panel from public sources.

Python 3.12 to 3.14 are supported and all tested in CI.

## Before opening a pull request

Run the full local gate; all four should be clean:

```bash
make test
ruff check research/pulseshift research/scripts research/tests
ruff format --check research/pulseshift research/scripts research/tests
mypy research/pulseshift --ignore-missing-imports
```

Then:

- If you changed the served model, run make model so model.json and model.js are committed in sync (the parity test guards this).
- If you changed an analysis, commit the regenerated tables and figures.

## Style

- Clean code, few comments: let names and structure carry the meaning.
- Keep functions short and single-purpose; push shared constants into pulseshift/config.py.
- Commit messages are short and human: one or two words describing what changed (e.g. "rerun tables", "review fixes", "attenuation bound").

## Scientific changes

The primary specification and the confirmatory/exploratory split are fixed in [paper/preregistration.md](paper/preregistration.md). New analyses are welcome, but please mark them as exploratory unless they were prespecified, and keep the leak-free, out-of-time discipline (climatology fit on training years only) intact: it is the project's central validity claim.
