# PLAN.md

Project scope, session arc, and roadmap context for the eval harness.
For working conventions (commands, markers, layout), see `CLAUDE.md`.

## What this project is

A pytest-based evaluation harness for LLM systems. Standalone — does not import from
or depend on any other project. Provider-abstracted so it can target any LLM endpoint,
but for this project it only ever targets Ollama (`llama3.2` via `localhost:11434`).

**One focused theme:** how do you score non-deterministic LLM output?

## Position in a larger roadmap

This is project 1 of 5 in an AI/QA learning portfolio. Each project gets its own repo
and a writeup. The full roadmap is at the bottom of this file. The order matters
because **scope decisions for this project depend on what's coming later** — see
"Out of scope" below.

**Project 1 (already complete):** https://github.com/sbezjak/llm-api-testing
FastAPI service proxying questions to local Ollama. 23 pytest tests. Hosted GitHub
Pages reports. `respx` for mocking, `ASGITransport` for in-process testing,
threshold assertions for non-determinism, marker split between fast/mocked and
`ollama`-integration. **That repo is the quality bar for this one** — same README
structure, same hosted-reports discipline, same marker conventions, same level
of polish. Do not import from it; just match its shape.

## Locked scope for this project

Do not expand this scope without explicit confirmation.

**Scope was deliberately cut at end of S4.** The original plan called
for 50 hand-written items + 200 LLM-generated items + several support
scripts. We cut both expansions and most of the scripts because the
10-item V1 set is already producing 5 distinct findings (~1 finding
per 2 items), and adding 240 more items would dilute signal-per-item
without teaching new concepts. Quality > size for an eval harness
writeup. The full reasoning is in `article.md` and the chat history
of S4. **If you (future agent) read this section and feel the urge
to "complete the original plan" by adding the 50-item expansion or
the LLM-generated dataset — don't.** The cut is the deliberate choice.

- **Golden set:** 18 hand-written question/expected-answer pairs in
  `data/golden_set.yaml` (10 main items already done, 4 calibration
  + 4 bias-swap to be added in S5).
- **Three scorers in the pytest harness (all done at end of S4):**
  1. Exact match — strict, unforgiving, the honest baseline.
  2. Semantic similarity — `sentence-transformers/all-MiniLM-L6-v2`,
     cosine, configurable threshold.
  3. LLM-as-judge — second Ollama call, hybrid rubric (correctness +
     relevance, 0–10 each, normalized to [0,1]), JSON parsing with
     regex fallback.
- **Disagreement test (centerpiece, done in S4):**
  `test_scorers_agree_on_verdict` runs semantic + judge on each item,
  fails when they disagree, logs both verdicts on every row.
- **Calibration test (S5):** "evaluate the evaluator." `data/human_labels.yaml`
  freezes the 10 golden-set Ollama outputs paired with the author's human
  verdict from `reports/human_eval_v2.md`. `tests/test_calibration.py`
  parametrizes (item × scorer) and asserts each scorer's PASS/FAIL agrees
  with the human verdict, with `xfail(strict=True)` on the items where
  Findings 1, 3, and 4 say the scorer structurally disagrees. The xfail
  set IS the contract: an unexpected pass means a finding has changed and
  article.md needs updating. This is closer to production-grade
  calibration than synthetic fixtures — real outputs, real labels,
  agreement-against-humans is the metric.
- **Bias sanity test (~4 items, S5):** demographic detail swap; the
  factual content of the answer shouldn't change. Sanity test, not
  a real bias evaluation.
- **Hosted pytest-html + coverage on GitHub Pages (S6),** same setup
  as project 1.
- **README at project-1 quality bar (S6).** Sections: what it
  teaches / what it tests / run it / project layout / tech.

**Cut from the original plan (do not re-add):**

- ~~50-item hand-written expansion~~ — 10 deliberately-adversarial
  items beat 50 mostly-easy ones for this writeup's purposes.
- **Deferred (not cut for the wrong reason):** the original plan called for
  `scripts/expand_dataset.py` to generate 200 synthetic Q&A pairs and
  validate them against the golden set for quality and diversity. An
  earlier version of this PLAN dismissed it as "ingestion engineering" —
  that was wrong. Synthetic test data validation IS eval-harness work
  (prompt design for synthesis, embedding-space coverage, scorer behavior
  on synthetic vs hand-written, the failure-mode question of whether LLMs
  generate items adversarial enough to break themselves). It teaches a
  distinct lesson Findings 1–6 don't cover. We're deferring it to an
  optional **S7** after S6 ships, not deleting it. The S5 (calibration +
  bias) story is coherent and complete on its own; wedging synthetic-data
  generation into S5 would dilute both. If S7 happens it produces a
  Finding 6/7 ("synthetic test data has these specific blind spots vs
  hand-written items") and a section in article.md and the README.
- ~~`scripts/scorer_comparison.py`~~ — partially redundant with the
  disagreement test, which already shows per-item scorer behavior in
  the html report.
- ~~`scripts/traditional_metrics_demo.py` for BLEU/ROUGE~~ — the point
  is already made in `article.md` Finding 1 (exact match fails on
  prose-wrapped right answers; BLEU/ROUGE share the same n-gram
  failure mode).

**Target test count:** ~35–40 tests total when S5 lands (currently
31 mocked + 41 ollama = 72; calibration adds ~12, bias adds ~4).

**Deferred ideas (not cut, not in scope right now):**

- **Determinism / temp=0 audit.** Run `OllamaProvider.generate()` 3-5×
  on the same prompt at temperature=0 and diff the outputs. They will
  not be identical. The lesson is that "temp=0" is a request, not a
  guarantee — which is *why* we freeze the calibration corpus instead
  of regenerating. Largely subsumed by `tests/test_judge_variance.py`.
- **Score-distribution visualization.** A small script that plots a
  histogram of every semantic and judge score in the V3 run, with
  right answers and wrong answers in different colors. Shows visually
  why no threshold can separate them — the visual proof behind the
  "semantic is not measuring correctness" finding. ~30 min of work,
  data already exists in test output. Optional polish if S6 finishes
  with budget left.
- **S7: synthetic test-data generation + validation.** See "Cut from
  the original plan" above — deferred, not deleted. Teaches a distinct
  lesson (synthetic data has these specific blind spots vs hand-written)
  that the current findings don't cover.

Beyond these, additional eval topics belong to later projects in the
roadmap (multi-turn eval, tool-use eval, RAG eval, agent eval,
cross-model benchmarking) — don't rope them into project 1.

## Out of scope

These are deliberately excluded because they belong to later projects in the
roadmap. Push back if asked to add them.

- Multiple LLM providers (project 5: model benchmarking)
- Adversarial / red-team prompts (project 3: red team test suite)
- Retrieval / RAG (project 2 in roadmap order: RAG + observability)
- Deep bias evaluation (BBQ benchmark, statistical tests across many runs)
- Web UI / dashboard
- CI pipeline (no GitHub Actions for now — one repo per project, one writeup
  per project, that's the rule)

## Session arc

Six focused sessions, mirroring how project 1 was built. Don't jump ahead.

1. **Scaffolding.** Repo structure, `pyproject.toml`, pytest config, marker
   registration, `Provider` ABC + `OllamaProvider`. Smoke test (mocked) +
   one live-Ollama test. End state: `uv run pytest -m "not ollama"` green
   in <1s, pushed to GitHub.
2. **Golden set v1 + exact match.** 10 hand-written items across 3
   difficulties. `ExactMatchScorer`. First parametrized test running the
   golden set through `OllamaProvider` and scoring with exact match.
3. **Semantic similarity scorer.** `sentence-transformers` integration,
   `SemanticScorer` with configurable cosine threshold. Parametrized test
   alongside exact match. Scorer disagreements become visible.
4. **LLM-as-judge scorer + disagreement test.** Rubric prompt, second
   Ollama call, structured output parsing. The disagreement test that
   compares all three scorers — this is the centerpiece. End state: you
   can see, per item, which scorer says what.
5. **Calibration + bias sanity (focused).** Add ~4 calibration items
   (known-correct + known-wrong, each scorer must score correctly) and
   ~4 bias-swap pairs (demographic detail swap, factual content
   shouldn't change) to `golden_set.yaml`. New tests:
   `test_calibration.py` and `test_bias.py`. Update `article.md` with
   whatever they find. Skip the dataset expansion (see "Cut from the
   original plan" above).
6. **Polish.** README at project-1 quality bar, hosted GitHub Pages
   reports, coverage, pytest durations, writeup draft. No support
   scripts beyond what already exists.

## Working principles

- **Explain the why behind decisions, not just the code.** This project
  is for learning. Skip pytest-101 — assume solid pytest/Python background.
  Go deeper on LLM-eval-specific reasoning.
- **Show code before writing it when the design has trade-offs.** Just
  write it when it's mechanical.
- **One session at a time.** Don't write session 3 code during session 1.
- **Push back on scope creep.** The scope above is locked deliberately.

---

# Appendix: full 5-project roadmap

For context on what comes after this project and why certain things are
out of scope.

## 1. Eval Harness ← THIS PROJECT

**Learn:** Semantic similarity (cosine), LLM-as-judge pattern, BLEU/ROUGE,
human eval rubrics, what "good enough" means for non-deterministic output,
how to build golden datasets, bias detection, edge case generation.

**Project:** Build a pytest-based eval harness. Create 30-50
question/expected-answer pairs across different difficulty levels and edge
cases. Run them through an LLM, then score responses three ways: exact
match, semantic similarity (using an embedding model), and LLM-as-judge
(a second model grades the first). Compare which scoring method catches
what. Then use an LLM to generate 200 more test cases and validate them
against your golden set for quality and diversity. The output is a
reusable framework you can point at any LLM system.

## 2. RAG System + Tests + Observability

**Learn:** Document chunking strategies, vector databases (ChromaDB is
easiest to start), embedding models, retrieval relevance, hybrid search,
RAG failure modes (wrong doc retrieved, answer not in context,
hallucinated sources). Also: Langfuse or LangSmith, tracing, token usage
tracking, latency monitoring, drift detection.

**Project:** Build a "test your own docs" RAG. Load 10-15 markdown or PDF
files, chunk them, embed into ChromaDB, query with an LLM. Write tests:
does it retrieve the right chunk? Does it answer correctly? Does it say
"I don't know" when the answer isn't in the docs? Does it hallucinate a
source that doesn't exist? Add Langfuse tracing to the whole thing. Run
the same 20 queries on day 1 and day 7, then compare: average latency,
token cost per query, response quality drift, any queries that suddenly
started failing. You get RAG testing and production monitoring in one
project.

## 3. Red Team Test Suite

**Learn:** Direct prompt injection, indirect injection (hidden
instructions in documents the AI reads), jailbreaks, data exfiltration,
system prompt extraction, privilege escalation, Unicode tricks,
multilingual attacks. Deeper OWASP Top 10 for LLMs beyond the overview
level.

**Project:** Take the existing FastAPI + LLM project (project 0, the
prerequisite) and try to break it. Build a structured adversarial test
suite in pytest: "ignore previous instructions," instructions hidden in
uploaded documents, attempts to extract the system prompt,
encoded/obfuscated attacks, prompts in unexpected languages, role-playing
attacks ("pretend you're a system admin"). Log what passes and what the
system catches. Categorize failures by severity. The end result is a
reusable red team test kit you can run against any LLM-powered API.

## 4. Agent Testing

**Learn:** ReAct pattern, tool calling, multi-step reasoning, error
handling in loops, how agents decide when to stop, failure modes
(infinite loops, wrong tool selection, cascading errors, partial
failures).

**Project:** Build a mini agent that has 3-4 tools (e.g., calculator,
weather API mock, file reader, web search mock). Then write tests that
focus on the decision chain, not just the final answer: does it pick the
right tool? Does it recover when a tool returns an error? Does it stop
after getting an answer or loop forever? What happens when two tools
return conflicting info? What if a tool is slow — does it time out
gracefully? Use pytest to assert on the full trace of tool calls. This is
the hardest project on the list but also the most forward-looking —
agentic systems are where everything is heading.

## 5. Model Benchmarking

**Learn:** Token pricing across providers, latency vs quality tradeoffs,
caching strategies (semantic caching), batching, when to use small vs
large models, prompt optimization for token efficiency.

**Project:** Take the eval harness from project 1 and run the same test
suite against multiple models — GPT-4o-mini, Claude Haiku, Claude Sonnet,
a local model via Ollama. Compare across three dimensions: cost per
query, latency, and quality score (from your eval framework). Find where
the cheap/fast model is good enough and where you actually need the
expensive one. Write a pytest report that recommends which model for
which use case.