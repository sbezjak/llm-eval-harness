# llm-eval-harness

Pytest-based evaluation harness for LLM systems. Targets a local Ollama
(`llama3.2`) backend. Explores how to score non-deterministic LLM output
five ways — exact match, BLEU, ROUGE-L, semantic similarity, and
LLM-as-judge — and where each scorer fails.

The point of this repo is not the code. It's the **10 findings** in
`article.md` about how LLM evaluation actually behaves: where each
scorer is wrong, when scorers disagree, and what those disagreements
teach you about scoring non-deterministic output.

For a public-facing companion that maps every concept to a
traditional-QA analogue and lays out the per-scorer failure-mode
matrix, see `docs/scoring-tradeoffs.md`.

## What's in here

| Path | Purpose |
|---|---|
| `eval_harness/scorers/` | The 5 scorers: `ExactMatchScorer`, `BleuScorer`, `RougeScorer`, `SemanticScorer`, `LLMJudgeScorer` |
| `eval_harness/providers/` | `OllamaProvider` HTTP adapter |
| `data/golden_set.yaml` | 10 hand-written question/expected pairs (the eval inputs) |
| `data/human_labels.yaml` | 10 frozen Ollama outputs + author's human verdict per item (calibration corpus) |
| `data/bias_pairs.yaml` | 4 paired questions for the bias-swap sanity check |
| `tests/` | The whole story (see below) |
| `article.md` | The 10 findings — read this if you read nothing else (private; gitignored in S6) |
| `docs/scoring-tradeoffs.md` | Public companion: QA-translation table, failure-mode matrix, deliberate trade-offs |
| `reports/human_eval_v2.md` | Hand-graded results from the V2 live run |
| `PLAN.md` | Scope, session arc, and what's deliberately out of scope |

## Setup

You need [`uv`](https://docs.astral.sh/uv/) and (for the live tests)
[Ollama](https://ollama.com/) running locally with `llama3.2` pulled.

```bash
# Install deps
uv sync

# Pull the model (one time, ~2GB)
ollama pull llama3.2

# Verify Ollama is up
curl -sf http://localhost:11434/api/tags | head
```

## Run the tests

Two markers split the suite:

- `mocked` — fast, no network. Mocks Ollama with `respx`. Runs in ~10s.
- `ollama` — hits a real local Ollama. Slow (~10 minutes for the full set).

```bash
# Fast: scorer unit tests, dataset checks, calibration on frozen outputs
uv run pytest -m "not ollama"

# Slow: live model + judge calls
uv run pytest -m ollama

# Everything
uv run pytest
```

Generate an HTML report (the kind of artifact a portfolio reviewer wants):

```bash
uv run pytest --html=reports/run.html --self-contained-html
```

## Run a specific test if you want to see one finding

Each finding in `article.md` has a corresponding test. Pick one:

```bash
# Finding 1 — exact match fails on prose-wrapped answers
uv run pytest tests/test_calibration.py::test_exact_match_calibration -v

# Finding 3 — semantic similarity fails on shape mismatch (e.g. factual_004)
uv run pytest tests/test_calibration.py::test_semantic_calibration -v

# Finding 4 + 8 — judge passes a hallucinated answer, AND it's stuck at 0.700
uv run pytest tests/test_calibration.py::test_judge_calibration -m ollama -v
uv run pytest tests/test_judge_variance.py -m ollama -v -s

# Finding 7 — bias-swap drift (David vs Priya)
uv run pytest tests/test_bias.py -m ollama -v -s

# Finding 9 — null result on length bias
uv run pytest tests/test_length_bias.py -m ollama -v -s

# Finding 10 — BLEU/ROUGE failure depends on reference shape, not output
uv run pytest tests/test_calibration.py::test_bleu_calibration tests/test_calibration.py::test_rouge_calibration -v

# The S4 centerpiece — scorers running side-by-side, where they disagree
uv run pytest tests/test_eval_pipeline.py::test_scorers_agree_on_verdict -m ollama -v -s
```

The `-s` flag is important on the live tests — it streams the per-item
print statements (question, expected, model output, scorer scores,
judge reasoning) to stdout. That's where most of the *evidence* lives;
the assertions are short.

## Reading the test output

Tests with `xfail(strict=True)` markers are not failures — they're
**documented disagreements**. Each one's `reason=` says in plain English
what the scorer does wrong on that item.

```
test_calibration.py::test_exact_match_calibration[factual_001]  XFAIL
  reason: exact match wrongly says FAIL: model returns the right answer
  wrapped in conversational prose, so output != expected (Finding 1)
```

If an `xfail` ever flips to `XPASS`, the suite breaks on purpose —
that means a documented finding has silently changed and we want to
re-investigate, not lose the finding.

## How the 10 findings map to tests

| Finding | What it shows | Test |
|---|---|---|
| 1 | Exact match fails on every prose-wrapped right answer | `test_calibration.py::test_exact_match_calibration` |
| 2 | A canonical paraphrase scores 0.725 — below a "reasonable" 0.75 | `test_semantic_scorer.py` |
| 3 | Semantic similarity is not measuring correctness | `test_calibration.py::test_semantic_calibration` |
| 4 | Self-grading bias passes the only wrong answer | `test_calibration.py::test_judge_calibration` |
| 5 | Judge reasoning text decouples from judge score | `test_eval_pipeline.py` (read the printed reasoning) |
| 6 | Calibration as a contract: xfail-as-documentation | the calibration tests, their xfail markers |
| 7 | Bias-swap drift on one of four pairs | `test_bias.py` |
| 8 | Judge isn't noisy at threshold — it's stuck at 0.700 | `test_judge_variance.py` |
| 9 | No detectable length bias on llama3.2 (a useful null) | `test_length_bias.py` |
| 10 | BLEU/ROUGE failure depends on reference shape, not output | `test_calibration.py::test_bleu_calibration`, `test_rouge_calibration` |

## A note on BLEU and ROUGE

BLEU and ROUGE are vocabulary-overlap metrics designed for translation
and summarization respectively — they count how many word-sequences
the model's output shares with a reference text. They are added in
S6.5 as the fourth and fifth scorers in the suite.

The headline result (Finding 10): on this corpus, both metrics fail on
the short-bare-reference items the same way exact match fails (`Paris`,
`Ag`, `8`), and both metrics *succeed* on the prose-vs-prose items
where reference and output share vocabulary in the same order
(`definition_002`, `reasoning_002`). Whether BLEU/ROUGE is a useful
metric on a given corpus depends on the **shape of the expected
references**, not on the metric's intrinsic quality. This pre-justifies
ROUGE-L as a reasonable starting metric for the RAG project (where
references are paragraphs) and pre-justifies *not* relying on it for
short-answer factual QA.

The xfail-strict mechanism caught a wrong prediction here: the initial
BLEU/ROUGE disagreement maps were over-broad and 5 calibration cases
went red on the first run as strict-XPASSes. Re-reading the corpus
sharpened the finding from "BLEU/ROUGE share exact match's failure
mode" to the dataset-shape claim above. See `article.md` Finding 10.

## The thing that surprised me most

Finding 8. I expected the judge's score on the threshold-edge item
(procedural_001 — Finding 4) to flicker between PASS and FAIL across
runs from temperature noise. Instead, 5 runs returned 5 identical 0.700.
The judge isn't flaky on its own hallucination — it's *consistently
wrong with the same exact score every single time*. Worse than
flakiness, because there's nothing for CI to catch. See `article.md`
Finding 8 for the production implications.

## Lint, format, etc.

```bash
uv run ruff check .
uv run ruff format .
```

## Out of scope (and why)

See `PLAN.md`. Short version: this repo is project 1 of 5; cross-model
benchmarking, RAG, red-team prompts, and agent testing each get their
own repo. Anything that would dilute "how do you score
non-deterministic LLM output" was deliberately cut.
