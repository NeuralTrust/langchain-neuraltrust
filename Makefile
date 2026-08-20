.PHONY: format lint typing test test-unit test-live install

install:
	uv sync --group test --group lint --group typing

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typing:
	uv run mypy langchain_neuraltrust

test-unit:
	uv run pytest tests/unit_tests

test-live:
	uv run pytest tests/integration_tests -m live

test: test-unit
