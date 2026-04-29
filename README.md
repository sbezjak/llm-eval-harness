# llm-eval-harness

Pytest-based evaluation harness for LLM systems. Targets a local Ollama (`llama3.2`) backend and explores how to score non-deterministic LLM output: exact match, semantic similarity, and LLM-as-judge — including where these scorers disagree.

> Work in progress. Scaffolding only. Full README lands in the final session.

## Run

```bash
uv sync
uv run pytest -m "not ollama"   # fast, mocked
uv run pytest -m ollama         # requires Ollama at localhost:11434
```

See `CLAUDE.md` for the full command reference and architecture intent.
