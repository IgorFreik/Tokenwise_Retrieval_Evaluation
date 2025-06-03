.PHONY: help install test lint format clean run pre-commit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	poetry install

test: ## Run tests
	poetry run pytest tests/ -v --cov=src

lint: ## Run linting checks
	poetry run flake8 src/
	poetry run mypy src/

format: ## Format code
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage

run: ## Run the main benchmarking script
	poetry run python src/main.py

pre-commit: ## Install pre-commit hooks
	poetry run pre-commit install

check: ## Run all checks (lint, test, format)
	make format
	make lint
	make test 