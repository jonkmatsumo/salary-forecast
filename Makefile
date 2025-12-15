.PHONY: help install install-dev setup test lint format format-check type-check check clean pre-commit security coverage run-api run-streamlit

# Default Python version
PYTHON := python3.12
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install development dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

setup: install-dev ## Install dev dependencies and set up pre-commit hooks
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linting checks (ruff)
	ruff check src/ tests/

lint-fix: ## Run linting checks and auto-fix issues
	ruff check --fix src/ tests/

format: ## Format code with black and isort
	black src tests
	isort src tests

format-check: ## Check code formatting without making changes
	black --check src tests
	isort --check-only src tests

type-check: ## Run type checking (mypy)
	mypy src/

check: format-check lint type-check ## Run all checks (format, lint, type-check)

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

security: ## Run security scanning (bandit and pip-audit)
	bandit -r src/ -x tests/ -s B101 -ll
	pip-audit --desc

clean: ## Clean up generated files and caches
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist/
	rm -rf build/

run-api: ## Run the FastAPI server
	uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000

run-streamlit: ## Run the Streamlit application
	streamlit run src/app/app.py

all: check test ## Run all checks and tests
