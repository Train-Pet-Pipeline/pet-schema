PYTHON := /Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python
PIP := /Users/bamboo/.miniconda3/envs/pet-pipeline/bin/pip

.PHONY: setup test lint clean

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
