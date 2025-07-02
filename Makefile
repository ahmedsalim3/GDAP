# Docker context includes: Makefile, pyproject.toml, uv.lock, src/, app/, docs/, tests/
.PHONY: help install install-dev run-script run-app run-app-docker run-app-docker-dev clean clean-docker test lint format

help:
	@echo "Available commands:"
	@echo ""
	@echo "  install          Install project dependencies using uv"
	@echo "  install-dev      Install development dependencies and pre-commit hooks"
	@echo "  fix              Run pre-commit hooks on all files"
	@echo "  test             Run tests with coverage report"
	@echo "  lint             Run ruff linter to check code quality"
	@echo "  format           Format code using ruff formatter"
	@echo "  run-script       Run the main Python script"
	@echo "  run-app          Run the Streamlit application locally"
	@echo "  run-app-docker   Build and run the application in Docker (production)"
	@echo "  run-app-docker-dev Build and run the application in Docker (development)"
	@echo "  clean            Clean up Python cache files and build artifacts"
	@echo "  clean-docker     Clean up Docker images and containers"
	@echo "  help             Show this help message"
	@echo ""

install:
	uv sync

install-dev:
	uv sync --all-extras
	uv run pre-commit install

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest tests/ --cov=src --cov=app --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

run-script:
	uv run python src/main.py

run-app:
	uv run streamlit run streamlit_app.py

run-app-docker:
	docker build -t gdap:latest .
	docker run --rm -p 8501:8501 gdap:latest

run-app-docker-dev:
	docker build -f Dockerfile.dev -t gdap:dev .
	docker run --rm -p 8501:8501 -v $(PWD):/app gdap:dev

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage*
	@echo "Done."

clean-docker:
	docker rm -f $(docker ps -aq) 2>/dev/null || true
	docker rmi -f gdap:latest gdap:dev 2>/dev/null || true
	docker system prune -f
	@echo "Done."

clean-docker-all:
	docker rm -f $(docker ps -aq) 2>/dev/null || true
	docker rmi -f gdap:latest gdap:dev 2>/dev/null || true
	docker rmi -f ghcr.io/astral-sh/uv:python3.12-bookworm-slim python:3.12-slim python:3.12-slim-bookworm 2>/dev/null || true
	docker system prune -af
	@echo "Done."
