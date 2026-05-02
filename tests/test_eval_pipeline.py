from __future__ import annotations

import pytest

from eval_harness.dataset import GoldenItem, load_golden_set
from eval_harness.providers.ollama import OllamaProvider
from eval_harness.scorers.base import Scorer
from eval_harness.scorers.exact_match import ExactMatchScorer
from eval_harness.scorers.judge import LLMJudgeScorer
from eval_harness.scorers.semantic import SemanticScorer
from tests._xfail import with_xfail

GOLDEN_ITEMS = load_golden_set()
GOLDEN_BY_ID = {i.id: i for i in GOLDEN_ITEMS}

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


# ---------------------------------------------------------------------------
# Documented disagreements on the LIVE pipeline.
#
# These dicts encode "scorer says FAIL on the live model answer for a
# structural reason we already understand." That's a different invariant
# than calibration's "scorer disagrees with human on a frozen output" —
# e.g. on procedural_001 exact match correctly says FAIL (human agrees,
# so it's not a calibration disagreement) but the pipeline assertion
# `result.passed` still fires, so we xfail it here too.
#
# strict=True: if a documented disagreement starts passing (e.g. the
# live model changes its answer shape), that's an XPASS and the suite
# breaks loudly. We want to know — it usually means model drift.
# ---------------------------------------------------------------------------

_PROSE = "exact match: model wraps the right answer in conversational prose, output != expected"

EXACT_MATCH_DISAGREEMENTS: dict[str, str] = {
    "factual_001": _PROSE,
    "factual_002": _PROSE,
    "factual_003": _PROSE,
    "factual_004": _PROSE,
    "definition_001": _PROSE,
    "definition_002": _PROSE,
    "procedural_001": (
        "exact match: model hallucinates a fake pytest flag; output != expected "
        "(scorer is correct here — same verdict as the human)"
    ),
    "reasoning_001": _PROSE,
    "reasoning_002": _PROSE,
    "reasoning_003": _PROSE,
}

SEMANTIC_DISAGREEMENTS: dict[str, str] = {
    "factual_002": (
        "semantic: expected is a 2-character symbol ('Ag'); embedding distance to a "
        "full sentence pulls cosine below 0.75"
    ),
    "factual_003": (
        "semantic: short expected ('South America') vs prose answer pulls cosine "
        "below 0.75"
    ),
    "factual_004": (
        "semantic: model returns a bulleted list of all 8 planets; expected is the "
        "digit '8' — embedding can't see past the shape mismatch"
    ),
    "definition_001": (
        "semantic: short expected ('Artificial Intelligence') vs full-sentence answer"
    ),
    "definition_002": (
        "semantic: same idea phrased with different words; embedding similarity "
        "below threshold"
    ),
    "procedural_001": (
        "semantic: model's hallucinated flag has different vocabulary from the "
        "real one; cosine below 0.75 (scorer is correct here — same verdict as "
        "the human)"
    ),
}

# Judge passes every live item in the golden set under llama3.2. If that
# ever changes (XPASS / hard fail in this test), it's a real signal.
JUDGE_DISAGREEMENTS: dict[str, str] = {}


def _live_params() -> list:
    """Cross-product of (item, scorer) with per-pair xfail markers."""
    scorers: list[tuple[Scorer, dict[str, str]]] = [
        (ExactMatchScorer(), EXACT_MATCH_DISAGREEMENTS),
        (SemanticScorer(threshold=0.75), SEMANTIC_DISAGREEMENTS),
        (LLMJudgeScorer(threshold=0.7), JUDGE_DISAGREEMENTS),
    ]
    params = []
    for item in GOLDEN_ITEMS:
        for scorer, dmap in scorers:
            param_id = f"{item.id}-{scorer.name}"
            if item.id in dmap:
                marks = [pytest.mark.xfail(strict=True, reason=dmap[item.id])]
            else:
                marks = []
            params.append(pytest.param(item, scorer, id=param_id, marks=marks))
    return params


@pytest.mark.ollama
@pytest.mark.parametrize("item,scorer", _live_params())
async def test_scorer_against_live_ollama(item: GoldenItem, scorer: Scorer):
    """End-to-end: ask a live Ollama, score with each scorer in turn.

    The pytest html report becomes the artifact: every (item, scorer) pair
    is a row. Disagreements between scorers — exact match fails on a
    prose-wrapped answer that semantic similarity passes — are exactly the
    signal we want to see, encoded as xfail-strict above.

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


# ---------------------------------------------------------------------------
# Centerpiece: do the two threshold-based scorers agree?
# ---------------------------------------------------------------------------

SEMANTIC_VS_JUDGE_DISAGREEMENTS: dict[str, str] = {
    "factual_002": (
        "semantic FAIL + judge PASS — expected is 'Ag', model returned a sentence; "
        "embedding can't bridge the shape gap, judge correctly recognises the "
        "answer is right"
    ),
    "factual_003": (
        "semantic FAIL + judge PASS — short expected ('South America') vs prose "
        "answer; same shape-gap pattern"
    ),
    "factual_004": (
        "semantic FAIL + judge PASS — bulleted list of planets vs the digit '8'"
    ),
    "definition_001": (
        "semantic FAIL + judge PASS — 'Artificial Intelligence' vs a full-sentence "
        "definition"
    ),
    "definition_002": (
        "semantic FAIL + judge PASS — same idea, different wording"
    ),
    "procedural_001": (
        "semantic FAIL + judge PASS — judge falls for its own hallucinated flag "
        "(self-grading bias); semantic correctly says FAIL"
    ),
}


@pytest.mark.ollama
@pytest.mark.parametrize("item_id", with_xfail(
    [i.id for i in GOLDEN_ITEMS], SEMANTIC_VS_JUDGE_DISAGREEMENTS,
))
async def test_scorers_agree_on_verdict(item_id: str):
    """
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
    reasoning) for each item — the agreements are data too. Documented
    disagreements are xfail-strict above; an unexpected agreement (XPASS)
    means a finding has silently changed and the suite will break.

    Don't "fix" disagreement failures by lowering thresholds; they are the
    data.
    """
    item = GOLDEN_BY_ID[item_id]
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

    assert semantic.passed == judge.passed, (
        f"\n  scorers disagree on {item.id!r}:"
        f"\n  question:  {item.question}"
        f"\n  expected:  {item.expected!r}"
        f"\n  got:       {output!r}"
        f"\n  semantic:  {sem_verdict} (score={semantic.score:.3f}, threshold=0.75)"
        f"\n  judge:     {jdg_verdict} (score={judge.score:.3f}, threshold=0.7)"
        f"\n  judge says: {judge.reason}"
    )
