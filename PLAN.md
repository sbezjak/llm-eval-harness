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

- **Golden set:** 50 hand-written question/expected-answer pairs in
  `data/golden_set.yaml`, across 3 difficulty levels, with ~10 tagged as edge cases.
  Edge case categories: `ambiguous`, `multi_part`, `no_good_answer`,
  `adversarial_looking_but_benign`, `underspecified`.
- **Three scorers in the pytest harness:**
  1. Exact match
  2. Semantic similarity — `sentence-transformers`, model `all-MiniLM-L6-v2`,
     cosine similarity with a configurable threshold
  3. LLM-as-judge — second Ollama call with a rubric prompt, returns a numeric
     score and a reasoning string
- **BLEU/ROUGE deliberately not in the test suite.** Instead a standalone
  `scripts/traditional_metrics_demo.py` runs them on the golden set, prints
  results, and the README explains why they're weak for open-ended LLM output.
- **Scorer disagreement test (centerpiece):** parametrized over the golden set,
  flags when the three scorers disagree by more than a configured threshold.
- **Calibration test (~4 items):** "evaluate the evaluator." Known-correct and
  known-wrong answers that each scorer must score correctly. Catches a broken
  scorer.
- **Bias sanity test (~4 items):** demographic detail swap (e.g. swap a name
  from one demographic to another), assert the factual content of the answer
  doesn't change. This is a sanity test, not a real bias evaluation.
- **LLM-generated 200-item expansion:** `scripts/expand_dataset.py` uses Ollama
  to generate ~200 additional items based on the golden 50, with a validation
  + dedup pass against the golden set. Output committed to
  `data/generated_set.yaml`.
- **`scripts/scorer_comparison.py`:** produces a markdown table to
  `reports/scorer_comparison.md` showing per-item scorer agreement plus summary
  stats. This is the screenshot for the writeup — make it look good.
- **Hosted pytest-html + coverage on GitHub Pages**, same setup as project 1.
- **README at project-1 quality bar.** Sections: what it teaches / what it
  tests / run it / project layout / tech.

**Target test count:** 30–35 tests total.

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
5. **Expand dataset + bias + calibration.** Golden set grows to 50
   hand-written items. `scripts/expand_dataset.py` produces 200 generated
   items with validation/dedup. Bias sanity test added. Calibration test
   added.
6. **Polish.** README, hosted GitHub Pages reports, coverage, durations,
   `scripts/scorer_comparison.py` produces the comparison report,
   `scripts/traditional_metrics_demo.py` for BLEU/ROUGE, writeup draft.

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