# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apache 2.0 `LICENSE` + `NOTICE` (`Copyright (c) 2026 Santander Group`).
- Community/governance files: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  `.github/SECURITY.md`, `CODEOWNERS`, `CITATION.cff`.
- SPDX headers (`Copyright (c) 2026 Santander Group` / `Apache-2.0`) on all
  Python source and test files.
- GitHub Actions workflows (third-party actions pinned to SHA digests):
  `ci`, `codeql`, `dep-scan`, `license-check`, `pattern-check`, `cla`,
  `stale`, `release`.
- `.github/dependabot.yml` (monthly Python + GitHub Actions updates),
  issue templates (bug, feature, config) and a PR template.
- `.github/pattern-check-allowlist.txt` for the internal-pattern scan.
- `black`/`mypy`/`pytest-cov` dev tooling and configuration for CI parity.

### Changed
- Project URLs and CI badge now point to `https://github.com/SantanderAI/auto-bayesian`.
- README opens with attribution to Santander AI Lab and project category.

## [0.1.0] - 2026-06-17

### Added
- Initial public release: relational materialization, deterministic
  preprocessing, three Bayesian-network candidates with automatic selection,
  F1-tuned threshold, persistence, evaluation, and Mermaid/CPD explainability.

[Unreleased]: https://github.com/SantanderAI/auto-bayesian/commits/main
[0.1.0]: https://github.com/SantanderAI/auto-bayesian/releases/tag/v0.1.0
