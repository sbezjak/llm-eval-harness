from __future__ import annotations

import pytest

from eval_harness.dataset import GoldenItem, load_golden_set
from eval_harness.providers.ollama import OllamaProvider
from eval_harness.scorers.exact_match import ExactMatchScorer

GOLDEN_ITEMS = load_golden_set()


@pytest.mark.ollama
@pytest.mark.parametrize("item", GOLDEN_ITEMS, ids=[i.id for i in GOLDEN_ITEMS])
async def test_exact_match_against_live_ollama(item: GoldenItem):
    """End-to-end: ask a live Ollama, score with strict exact match.

    Most items are expected to FAIL — that's the lesson, not a bug. Strict
    exact match cannot tolerate the prose wrapping that real LLMs produce
    ("The capital of France is Paris." vs "Paris"). The pytest html report
    becomes the artifact: scan the failures and you can see why we need
    semantic similarity (session 3) and LLM-as-judge (session 4).
    """
    provider = OllamaProvider(model="llama3.2", timeout=180.0, temperature=0.0)
    output = await provider.generate(item.question)

    result = ExactMatchScorer().score(item.question, output, item.expected)

    # Captured by pytest and embedded in the html report for both passed and
    # failed runs — the readable artifact for the writeup.
    print(f"\nquestion: {item.question}")
    print(f"expected: {item.expected!r}")
    print(f"got:      {output!r}")
    print(f"score:    {result.score} ({result.reason})")

    assert result.passed, (
        f"\n  question: {item.question}"
        f"\n  expected: {item.expected!r}"
        f"\n  got:      {output!r}"
        f"\n  reason:   {result.reason}"
    )
