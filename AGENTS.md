# Repo Rules

## Intent

This repo builds a small framework for Bayesian-network classification from
relational tabular data. Keep the code easy to read, explicit, and boring.

## Design Rules

- Prefer direct code over abstraction layers.
- Each module should have one clear responsibility.
- Keep public APIs typed and small.
- Keep defaults deterministic.
- Avoid hidden behavior and side effects.
- Add a dependency only when it removes real complexity.

## Structure

- `schema.py` owns config parsing and validation.
- `materialize.py` owns relational joins and aggregates.
- `preprocess.py` owns deterministic discretization and category handling.
- `engine_pgmpy.py` is the only place allowed to call `pgmpy`.
- `automl.py` owns candidate training and model selection.
- `model.py` owns persistence and prediction.
- `cli.py` stays thin.

## Commands

- Install: `uv sync --dev`
- Lint: `uv run ruff check .`
- Test: `uv run pytest`
- Train sample: `uv run auto-bayesian train examples/lead_scoring.toml`
