from __future__ import annotations

import pytest

from eval_harness.dataset import GoldenItem, load_golden_set
from eval_harness.providers.ollama import OllamaProvider
from eval_harness.scorers.base import Scorer
from eval_harness.scorers.exact_match import ExactMatchScorer
from eval_harness.scorers.semantic import SemanticScorer

GOLDEN_ITEMS = load_golden_set()
SCORERS: list[Scorer] = [ExactMatchScorer(), SemanticScorer(threshold=0.75)]

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
    signal we want to see. The judge scorer joins this matrix in session 4.
    """
    output = await _ollama_response(item)
    result = scorer.score(item.question, output, item.expected)

    print(f"\nscorer:   {scorer.name}")
    print(f"question: {item.question}")
    print(f"expected: {item.expected!r}")
    print(f"got:      {output!r}")
    print(f"score:    {result.score} ({result.reason})")

    assert result.passed, (
        f"\n  scorer:   {scorer.name}"
        f"\n  question: {item.question}"
        f"\n  expected: {item.expected!r}"
        f"\n  got:      {output!r}"
        f"\n  reason:   {result.reason}"
    )
