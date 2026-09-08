.PHONY: install install-dev setup deploy format typecheck test openapi

export UV_FROZEN := 1

install:
	uv sync

install-dev:
	uv sync --extra dev

setup:
	uv run modal run setup_artifacts.py

deploy:
	uv run modal deploy app.py

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run pyright

test:
	uv run pytest

openapi:
	uv run python -m scripts.make_openapi
