.PHONY: setup lint test demo
setup:
	uv venv && uv pip install -e ".[dev]"
lint:
	uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
test:
	uv run pytest
demo:
	uv run newsalpha run --config configs/synthetic.yaml
