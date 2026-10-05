# Contributing

## Setup and checks

Python 3.12+ and Node 22; from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r research/requirements.txt -r research/requirements-dev.txt
PYTHONPATH=research pytest research/tests --cov=pulseshift --cov-branch --cov-fail-under=60
node --test tests/app.test.cjs
ruff check research
ruff format --check research
mypy research/pulseshift --ignore-missing-imports
```

Dependencies are pinned. Test changed behavior; keep commit titles short.

## Reproduce

```bash
cd research
(cd data/processed && shasum -a 256 -c panel.sha256)
PYTHONPATH=. python scripts/run_analysis.py
PYTHONPATH=. python scripts/train_model.py
PYTHONPATH=. python scripts/validate_seoul.py
```

Analysis uses the committed panel; Seoul needs its public UCI archive on first use. Tables and figures go to `outputs/`; the app uses `model.js`. Regenerate both results and coefficients after feature changes. Keep source hashes, panel checksums, versions and split metadata.

`PYTHONPATH=. python scripts/build_data.py` downloads the public source data. Raw and interim caches stay untracked. Constants and source URLs are in `research/pulseshift/config.py`; verify timestamps and coverage before replacing the panel.

## Preview

From repository root, run `python -m http.server 8000` and open `http://localhost:8000`. Weather and AQI requests need network access; manual inference runs locally.

## Research and security

Fit labels, scalers and models on training data; hold out whole dates when evaluating the Seoul method. Bike-share demand is an aggregate proxy, not individual activity or a health outcome. Policy recovery is a simulated scenario; dataset terms apply separately from MIT.

Report vulnerabilities through [private advisories](https://github.com/urmeo/PulseShift/security/advisories/new) with version, reproduction and impact. Omit secrets and personal data; keep details private until fixed. Treat contributors respectfully.
