# Contributing to Knowledge-Agent

Thank you for helping improve Knowledge-Agent. Contributions of code,
documentation, tests, evaluation data and operational guidance are welcome.

## Before you start

- Search existing issues and pull requests.
- Open an issue before a large architectural change.
- Never include private documents, production logs, model weights, credentials
  or personal data in an issue, commit or test fixture.
- Security vulnerabilities must follow [SECURITY.md](SECURITY.md).

## Development setup

Requirements: Python 3.11–3.13, `uv`, Git and Docker Compose.

```bash
git clone <your-fork-url>
cd Knowledge-Agent
uv sync --frozen --group dev
```

Create a local configuration only when required:

```bash
cp .env.example .env
```

`.env` is ignored. Use synthetic values and data for development.

## Quality checks

Run the checks relevant to your change before opening a pull request:

```bash
uv run --frozen ruff check .
uv run --frozen mypy app eval scripts
uv run --frozen pytest -q
uv run --frozen python scripts/ci/check_secrets.py
docker compose --env-file deploy/compose.env.example config --quiet
```

Tests that require external infrastructure must use the `integration` marker
and remain opt-in.

## Pull requests

- Keep each pull request focused on one outcome.
- Add or update tests for behavior changes.
- Update documentation and examples when configuration or APIs change.
- Preserve tenant and owner scoping in every data access path.
- Describe migration and rollback steps for schema or deployment changes.
- Use clear commit messages, for example `feat: add parser adapter` or
  `fix: preserve readiness during CPU inference`.

By submitting a contribution, you agree that it is licensed under the Apache
License 2.0 and that you have the right to submit it.
