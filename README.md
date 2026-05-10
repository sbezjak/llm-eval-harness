# llm-eval-harness

Live reports: [test report](https://sbezjak.github.io/llm-eval-harness/reports/report.html) · [coverage](https://sbezjak.github.io/llm-eval-harness/reports/coverage/)

Built as project 1 of 5 exploring AI/LLM testing. A writeup is in progress.

A pytest-based evaluation harness for LLM systems. It runs a fixed golden set
through a local model, scores the outputs with five different scorers, and
encodes each scorer's known limitations as `xfail(strict=True)` tests so
silent regressions break the build.

The harness targets [Ollama](https://ollama.com/) at `localhost:11434` as the
model backend. Python 3.11+, managed with [`uv`](https://docs.astral.sh/uv/).

## Overview

The project answers one question: *how do you assert correctness on LLM
output when there is no canonical "right" wording?* Five scorers are
implemented and calibrated against a 10-item golden set with human-graded
labels:

| Scorer | Signal |
|---|---|
| `ExactMatchScorer` | `output == expected` |
| `BleuScorer` | `sacrebleu` n-gram overlap (BLEU-4) |
| `RougeScorer` | `rouge-score` ROUGE-L F1 |
| `SemanticScorer` | `sentence-transformers` cosine similarity |
| `LLMJudgeScorer` | second Ollama call with a hybrid correctness/relevance rubric |

Each scorer implements the same async contract:

```python
class Scorer:
    async def score(self, question: str, output: str, expected: str) -> ScoreResult: ...
```

Findings from the calibration run (95 passed, 51 xfailed, 98% coverage) are
documented as executable tests, see `docs/background.md` for the
per-scorer failure-mode matrix and a plain-language walkthrough.

## Architecture

```
eval_harness/
├── providers/       # backend adapters; only place that issues HTTP
│   └── ollama.py
├── scorers/         # pure (question, output, expected) -> ScoreResult
│   ├── exact_match.py
│   ├── bleu.py
│   ├── rouge.py
│   ├── semantic.py
│   └── llm_judge.py
└── dataset.py       # YAML loader + pydantic validation
docs/
└── background.md       # QA-engineer walkthrough, failure-mode matrix, trade-offs
data/
├── golden_set.yaml      # 10 question/expected pairs
├── human_labels.yaml    # frozen model outputs + human PASS/FAIL grades
└── bias_pairs.yaml      # name-swap pairs for bias drift checks
tests/                   # mocked + ollama-marked test suites
```

Architectural invariants:

- HTTP I/O lives only in `providers/`. Tests mock at this boundary with
  `respx`. Scorers are pure and unit-testable without network access.
- The package layout is pinned in `pyproject.toml`
  (`packages = ["eval_harness"]`), do not relocate.
- `httpx` is the transport used by `OllamaProvider` and mocked by `respx`.
  No FastAPI surface exists or is planned for this project.

## Requirements

- Python ≥ 3.11
- `uv` for environment management
- Ollama running at `localhost:11434` with the `llama3.2` model pulled
  (only required for `@pytest.mark.ollama` tests)

## Installation

```bash
uv sync                       # installs runtime + dev dependencies
ollama pull llama3.2          # ~2 GB, one-time
curl -sf http://localhost:11434/api/tags | head    # health check
```

## Testing

The suite is split by two custom markers (defined in `pyproject.toml`):

| Marker | Purpose | Runtime |
|---|---|---|
| `mocked` | `respx`-mocked Ollama; scorer logic, dataset validation, calibration on frozen outputs | ~10 s |
| `ollama` | live model + judge calls | ~7 min |

`asyncio_mode = "auto"` is set, so async tests do not need
`@pytest.mark.asyncio`.

```bash
uv run pytest                          # full suite
uv run pytest -m "not ollama"          # fast path, no network
uv run pytest -m ollama                # live model only
uv run pytest -m mocked                # mocked unit tests only
uv run pytest tests/path/to/test.py::test_name
```

### `xfail(strict=True)` as executable documentation

Every known scorer limitation is encoded as an `xfail(strict=True)` test
with a `reason` that names the finding. Example:

```
test_calibration.py::test_exact_match_calibration[factual_001]  XFAIL
  reason: exact match wrongly says FAIL: model returns the right answer
  wrapped in conversational prose, so output != expected (Finding 1)
```

If a scorer ever silently *stops* failing on that item, strict mode flips
the test red and forces investigation. The xfail set is the spec for what
the scorers are known to get wrong.

### Coverage and HTML report

```bash
uv run pytest \
  --html=reports/report.html --self-contained-html \
  --cov=eval_harness --cov-report=html:reports/coverage
```

Hosted artifacts from the latest run:

- Pytest HTML: https://sbezjak.github.io/llm-eval-harness/reports/report.html
- Coverage:    https://sbezjak.github.io/llm-eval-harness/reports/coverage/

## Findings

Ten findings are encoded as tests. Use them as entry points into the codebase.

| # | Finding | Test |
|---|---|---|
| 1 | Exact match fails on prose-wrapped answers | `test_calibration.py::test_exact_match_calibration` |
| 2 | A 0.75 cosine threshold rejects right answers that score 0.725 | `test_semantic_scorer.py` |
| 3 | Semantic-similarity score distributions for right/wrong answers overlap, no separating threshold | `test_calibration.py::test_semantic_calibration` |
| 4 | Self-grading bias: the judge passes the model's own hallucinations | `test_calibration.py::test_judge_calibration` |
| 5 | The judge's written reasoning can contradict its numeric score | `test_eval_pipeline.py` (`-s` to read prints) |
| 6 | `xfail(strict=True)` turns the suite into a tripwire for silent scorer drift | every `xfail` in `test_calibration.py` |
| 7 | Bias-swap (David vs. Priya) detects output drift on 1 of 4 paired prompts | `test_bias.py` |
| 8 | Judge variance is zero at the threshold, stuck at 0.700 across 5 runs | `test_judge_variance.py` |
| 9 | No detectable length bias on `llama3.2` (null result) | `test_length_bias.py` |
| 10 | BLEU/ROUGE viability depends on reference-text shape, not the metric | `test_calibration.py::{test_bleu_calibration,test_rouge_calibration}` |

Pass `-s` on the `ollama`-marked tests to stream per-item evidence
(question, model output, per-scorer scores, judge reasoning) to stdout.

## Development

```bash
uv run ruff check .
uv run ruff format .
```

Ruff is configured for `line-length = 100` and `target-version = "py311"`.

## Further reading

- `docs/background.md`, LLM-eval concepts in plain pytest words, a
  per-scorer failure-mode matrix, and the deliberate trade-offs.
- `CLAUDE.md`, guidance for AI assistants working in this repo.
