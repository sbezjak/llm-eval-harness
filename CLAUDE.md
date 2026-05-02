# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pytest-based evaluation harness for LLM systems. Targets Ollama (local, `localhost:11434`) as the model backend. Python 3.11+, managed with `uv`.

Three scorers live under `eval_harness/scorers/`: `ExactMatchScorer`, `SemanticScorer` (sentence-transformers, cosine), `LLMJudgeScorer` (second Ollama call with a hybrid rubric prompt). All three implement the same async `Scorer.score(question, output, expected) -> ScoreResult` contract. Provider code is in `eval_harness/providers/` (only `OllamaProvider` so far). Golden set lives at `data/golden_set.yaml`, loaded and validated by `eval_harness/dataset.py`. Do not relocate the package layout — `pyproject.toml` pins `packages = ["eval_harness"]` for the wheel build.

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

## Working style with this user

- **Prepare drafts/templates for any task the user has to do by hand.** When the next step is something only the user can do (write golden-set items, grade outputs, decide which threshold feels right), prepare a fill-in-the-blanks file with the structure pre-built — don't make the user start from a blank page. Examples: a markdown table with rows pre-filled from data, a YAML scaffold with TODOs, a notes template with section headings. Reduce the user's task to filling in the squishy parts.
- **Capture explanations to `article.md` when teaching.** When the user asks "explain this to me" and the answer is non-trivial, mirror it (lightly cleaned up) into `article.md` as reference material. The chat scrolls; the article stays.
- **Default to production / best-practice solutions; take the pragmatic shortcut only when the trade-off is justified for this project's scope — and call the trade-off out explicitly.** Don't silently pick the easy path. Name what the production-grade pattern would be, name why we're not doing it here (scope, runtime context, test budget), and write the trade-off into `article.md` so the writeup shows it was a deliberate choice, not an oversight. Worked example: scorers are async-only because this harness has one runtime context (pytest); production frameworks like DeepEval/Ragas use dual sync+async interfaces to avoid async contagion. We picked the simpler path eyes-open — see `article.md`.

