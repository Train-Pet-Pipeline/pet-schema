-include ../pet-infra/shared/Makefile.include

PYTHON := /Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python
PIP := /Users/bamboo/.miniconda3/envs/pet-pipeline/bin/pip

.PHONY: setup test lint clean

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short
