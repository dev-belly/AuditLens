# AuditLens - developer entry points.
#
# Every target is a thin wrapper around a command you can also run by hand; the
# point of the Makefile is that a reviewer can reproduce the whole project without
# reading the source first.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help install install-dev pipeline generate clean test test-fast coverage \
        dashboard sql lint verify

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install runtime and development dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt

pipeline:  ## Run the full nine-stage analytics pipeline
	$(PYTHON) src/run_pipeline.py

generate:  ## Regenerate the synthetic ledger only
	$(PYTHON) -c "from src.data_generator import generate_dataset; generate_dataset()"

sql:  ## Build the SQLite warehouse and run the analyst queries
	$(PYTHON) src/database.py

dashboard:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py

test:  ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-fast:  ## Run everything except the dashboard render tests
	$(PYTHON) -m pytest tests/ -q --ignore=tests/test_dashboard.py

coverage:  ## Run the tests with a coverage report
	$(PYTHON) -m pytest tests/ --cov=src --cov=dashboard --cov-report=term-missing

verify:  ## Run the pipeline from scratch, then the tests
	$(MAKE) pipeline
	$(MAKE) test

clean:  ## Remove generated data, charts and caches (keeps the source)
	rm -rf data/raw data/processed data/*.db outputs/charts outputs/reports
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
