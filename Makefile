.PHONY: all setup data analysis model seoul test lint clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUN := cd research && PYTHONPATH=. ../$(PY)

all: setup analysis model test

setup: $(VENV)/.installed

$(VENV)/.installed: research/requirements.txt research/requirements-dev.txt
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r research/requirements.txt -r research/requirements-dev.txt
	touch $@

data: setup
	$(RUN) scripts/build_data.py

analysis: setup
	$(RUN) scripts/run_analysis.py

model: setup
	$(RUN) scripts/train_model.py

seoul: setup
	$(RUN) scripts/validate_seoul.py

test: setup
	$(RUN) -m pytest tests/ -q

lint: setup
	$(VENV)/bin/ruff check research/pulseshift research/scripts research/tests
	$(VENV)/bin/ruff format --check research/pulseshift research/scripts research/tests
	$(VENV)/bin/mypy research/pulseshift --ignore-missing-imports

clean:
	rm -rf $(VENV) research/data/raw research/data/interim
	find research -name __pycache__ -type d -exec rm -rf {} +
