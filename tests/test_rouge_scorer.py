from __future__ import annotations

import pytest

from eval_harness.scorers.rouge import RougeScorer


async def test_identical_strings_score_at_one():
    r = await RougeScorer().score("q", "The capital of France is Paris.", "The capital of France is Paris.")
    assert r.passed is True
    assert r.score == pytest.approx(1.0, abs=1e-6)


async def test_unrelated_strings_score_low():
    r = await RougeScorer().score(
        "q", "Water boils at 100 degrees Celsius.", "The capital of France is Paris."
    )
    assert r.passed is False
    assert r.score < 0.3


async def test_prose_wrapped_short_answer_scores_below_threshold():
    """ROUGE-L F1 on a 1-word reference vs a full sentence: recall is
    1.0 (the reference word appears in the output), but precision is
    1/N where N is the output length, so F1 collapses on prose-wrapping
    just like BLEU and exact match.
    """
    r = await RougeScorer().score(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Paris",
    )
    assert r.passed is False
    assert r.score < 0.40


async def test_partial_overlap_scores_in_middle():
    """Some shared vocabulary, some not — ROUGE-L should sit in the middle."""
    r = await RougeScorer().score(
        "q",
        "The cat sat quietly on the warm mat",
        "The cat sat on the mat",
    )
    assert 0.3 < r.score < 1.0


async def test_score_is_in_unit_interval():
    r = await RougeScorer().score("q", "alpha beta gamma", "delta epsilon zeta")
    assert 0.0 <= r.score <= 1.0


async def test_threshold_gates_verdict():
    output = "the cat sat on the mat"
    expected = "the cat sat on the mat"
    lenient = await RougeScorer(threshold=0.1).score("q", output, expected)
    strict = await RougeScorer(threshold=0.99).score("q", output, expected)
    assert lenient.passed is True
    assert lenient.score == pytest.approx(strict.score, abs=1e-6)


async def test_reason_mentions_threshold_and_variant():
    r = await RougeScorer().score("q", "Paris", "Paris")
    assert "threshold" in r.reason
    assert "rougeL" in r.reason


def test_threshold_validated_at_construction():
    with pytest.raises(ValueError):
        RougeScorer(threshold=1.5)
    with pytest.raises(ValueError):
        RougeScorer(threshold=-0.1)
