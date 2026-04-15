-include ../pet-infra/shared/Makefile.include

PYTHON := python
PIP := python -m pip

.PHONY: setup test lint clean

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short
