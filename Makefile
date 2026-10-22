.PHONY: setup lint test labels

PYTHON := .venv/Scripts/python
RUFF := .venv/Scripts/ruff
MYPY := .venv/Scripts/mypy
PYTEST := .venv/Scripts/pytest

setup:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check src/chilefuego tests
	$(MYPY) src/chilefuego

test:
	$(PYTEST)

labels:
	$(PYTHON) -m chilefuego.labels.pipeline --config configs/config.yaml
