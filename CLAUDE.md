# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pytest-based evaluation harness for LLM systems. Targets Ollama (local, `localhost:11434`) as the model backend. Python 3.11+, managed with `uv`.

As of this writing the repository is a scaffold: `eval_harness/`, `eval_harness/providers/`, and `eval_harness/scorers/` exist as empty packages; `data/`, `reports/`, `scripts/`, and `tests/` are empty. Implementations should be added under these directories — do not relocate the package layout, since `pyproject.toml` pins `packages = ["eval_harness"]` for the wheel build.

## Commands

Use `uv` for all environment and execution tasks:

- Install / sync deps (including dev group): `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Run only fast (mocked) tests: `uv run pytest -m mocked`
- Run only tests that hit a real Ollama: `uv run pytest -m ollama` (requires Ollama running at `localhost:11434`)
- Skip Ollama tests: `uv run pytest -m "not ollama"`
- Coverage / HTML report: `uv run pytest --cov=eval_harness --html=reports/pytest.html`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`

## Test conventions (configured in pyproject.toml)

- `asyncio_mode = "auto"` — async tests do not need `@pytest.mark.asyncio`.
- Two custom markers gate Ollama-dependent tests:
  - `@pytest.mark.ollama` — slow, requires a live Ollama instance.
  - `@pytest.mark.mocked` — uses `respx` to mock the Ollama HTTP API; should be the default for unit tests.
- `testpaths = ["tests"]`; `ruff` line length is 100, target `py311`.

## Architecture intent

The package layout signals the intended seams; preserve them when adding code:

- `eval_harness/providers/` — adapters that talk to model backends (Ollama first). Provider code is the only place that should issue HTTP calls; tests mock at this boundary with `respx`.
- `eval_harness/scorers/` — pure scoring functions over (input, output, expected) tuples. `sentence-transformers` is a dep, so semantic-similarity scorers are expected here. Keep scorers free of I/O so they can be unit-tested without the `ollama`/`mocked` markers.
- `data/` — eval datasets (likely YAML, given the `pyyaml` dep).
- `reports/` — generated pytest/coverage output; treat as build artifacts.
- `scripts/` — CLI entry points / one-off runners.

`fastapi` and `httpx` are declared deps, suggesting a future HTTP surface (either the harness exposes results via FastAPI, or a fixture spins up a FastAPI app to evaluate). No such code exists yet — confirm intent with the user before adding either.
