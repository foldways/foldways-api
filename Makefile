.PHONY: install install-dev setup deploy deploy-custom format typecheck test openapi

export UV_FROZEN := 1

install:
	uv sync

install-dev:
	uv sync --extra dev

setup:
	uv run modal run setup_artifacts.py

deploy:
	uv run modal deploy app.py

deploy-custom:
	@[ -f .env ] || { echo "deploy-custom needs a .env file. Copy .env.example to .env."; exit 1; }
	set -a; \
	. ./.env; \
	set +a; \
	$(MAKE) deploy

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run pyright

test:
	uv run pytest

openapi:
	uv run python -m scripts.make_openapi
