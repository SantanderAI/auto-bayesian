# Contributing to auto-bayesian

Thanks for your interest in improving `auto-bayesian`. This project values small,
readable, explicit code. Contributions that keep it that way are very welcome.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
uv sync --dev
```

## Workflow

1. Fork the repository and create a feature branch.
2. Make your change, keeping modules focused and public APIs typed.
3. Run the checks below before opening a pull request.
4. Open a pull request describing the motivation and the change.

## Checks

```bash
uv run ruff check .                                  # lint
uv run pytest                                        # tests
uv run auto-bayesian train examples/lead_scoring.toml  # smoke test
```

Please add or update tests for any behavior change.

## Design principles

These guidelines keep the codebase approachable:

- Prefer direct code over abstraction layers.
- Each module has one clear responsibility.
- Keep public APIs typed and small, and defaults deterministic.
- Avoid hidden behavior and side effects.
- Add a dependency only when it removes real complexity.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
