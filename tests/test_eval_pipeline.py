from __future__ import annotations

import pytest

from eval_harness.dataset import GoldenItem, load_golden_set
from eval_harness.providers.ollama import OllamaProvider
from eval_harness.scorers.base import Scorer
from eval_harness.scorers.exact_match import ExactMatchScorer
from eval_harness.scorers.judge import LLMJudgeScorer
from eval_harness.scorers.semantic import SemanticScorer

GOLDEN_ITEMS = load_golden_set()
SCORERS: list[Scorer] = [
    ExactMatchScorer(),
    SemanticScorer(threshold=0.75),
    LLMJudgeScorer(threshold=0.7),
]

# Cache Ollama responses across scorers: each item's question is asked
# exactly once per pytest run, no matter how many scorers we evaluate it
# with. With temperature=0 the call is effectively deterministic anyway,
# but caching keeps the live suite from doing N_items * N_scorers HTTP
# round-trips for no reason.
_response_cache: dict[str, str] = {}


async def _ollama_response(item: GoldenItem) -> str:
    if item.id not in _response_cache:
        provider = OllamaProvider(model="llama3.2", timeout=180.0, temperature=0.0)
        _response_cache[item.id] = await provider.generate(item.question)
    return _response_cache[item.id]


@pytest.mark.ollama
@pytest.mark.parametrize("scorer", SCORERS, ids=[s.name for s in SCORERS])
@pytest.mark.parametrize("item", GOLDEN_ITEMS, ids=[i.id for i in GOLDEN_ITEMS])
async def test_scorer_against_live_ollama(item: GoldenItem, scorer: Scorer):
    """End-to-end: ask a live Ollama, score with each scorer in turn.

    The pytest html report becomes the artifact: every (item, scorer) pair
    is a row. Disagreements between scorers — exact match fails on a
    prose-wrapped answer that semantic similarity passes — are exactly the
    signal we want to see.

    For the judge specifically, the *rendered rubric prompt* is logged so
    you can confirm what the judge actually saw. Useful when debugging
    judge-reasoning artifacts (e.g. systematic "did not directly address
    the question" rationalizations) — sometimes the issue is in how the
    prompt template substitutes multi-line `got` text, not in the model.
    """
    output = await _ollama_response(item)
    result = await scorer.score(item.question, output, item.expected)

    print(f"\nitem:     {item.id}")
    print(f"scorer:   {scorer.name}")
    print(f"question: {item.question}")
    print(f"expected: {item.expected!r}")
    print(f"got:      {output!r}")
    print(f"score:    {result.score} ({result.reason})")
    if isinstance(scorer, LLMJudgeScorer):
        print("\n--- rubric prompt sent to judge ---")
        print(scorer.build_prompt(item.question, output, item.expected))
        print("--- end rubric prompt ---")

    assert result.passed, (
        f"\n  scorer:   {scorer.name}"
        f"\n  question: {item.question}"
        f"\n  expected: {item.expected!r}"
        f"\n  got:      {output!r}"
        f"\n  reason:   {result.reason}"
    )


@pytest.mark.ollama
@pytest.mark.parametrize("item", GOLDEN_ITEMS, ids=[i.id for i in GOLDEN_ITEMS])
async def test_scorers_agree_on_verdict(item: GoldenItem):
    """The S4 centerpiece: do the two threshold-based scorers agree on
    pass/fail per item?

    Exact match is the strict baseline — it fails on essentially every
    prose-wrapped answer, so we deliberately exclude it from the agreement
    check. The interesting question is whether the two *useful* scorers
    (semantic similarity, LLM-as-judge) agree. They measure different
    things: semantic measures textual proximity, judge measures whether
    the question was answered. When they disagree, that's the project's
    signal:

    - semantic FAIL + judge PASS → semantic was too strict (right answer,
      wrong shape — e.g. the "8 planets" bulleted list).
    - semantic PASS + judge FAIL → semantic was fooled by topical
      vocabulary; judge caught a wrong answer (e.g. a confident
      hallucination that uses the right keywords).

    Per-item logs print on every run regardless of agreement, so the html
    report shows the full picture (both verdicts, both scores, the judge's
    reasoning) for each item — the agreements are data too. The test
    itself only fails on disagreement; failure messages are deliberately
    redundant with the logs because pytest surfaces them in the failure
    table at the top of the report.

    Don't "fix" disagreement failures by lowering thresholds; they are the
    data.
    """
    output = await _ollama_response(item)

    semantic = await SemanticScorer(threshold=0.75).score(item.question, output, item.expected)
    judge = await LLMJudgeScorer(threshold=0.7).score(item.question, output, item.expected)

    sem_verdict = "PASS" if semantic.passed else "FAIL"
    jdg_verdict = "PASS" if judge.passed else "FAIL"
    agree = "AGREE" if semantic.passed == judge.passed else "DISAGREE"

    print(f"\nitem:      {item.id}  [{agree}]")
    print(f"question:  {item.question}")
    print(f"expected:  {item.expected!r}")
    print(f"got:       {output!r}")
    print(f"semantic:  {sem_verdict}  (score={semantic.score:.3f}, threshold=0.75)")
    print(f"judge:     {jdg_verdict}  (score={judge.score:.3f}, threshold=0.7)")
    print(f"judge says: {judge.reason}")

    if semantic.passed != judge.passed:
        pytest.fail(
            f"\n  scorers disagree on {item.id!r}:"
            f"\n  question:  {item.question}"
            f"\n  expected:  {item.expected!r}"
            f"\n  got:       {output!r}"
            f"\n  semantic:  {sem_verdict} (score={semantic.score:.3f}, threshold=0.75)"
            f"\n  judge:     {jdg_verdict} (score={judge.score:.3f}, threshold=0.7)"
            f"\n  judge says: {judge.reason}"
        )
