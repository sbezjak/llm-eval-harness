from __future__ import annotations

import pytest

from eval_harness.scorers.bleu import BleuScorer


async def test_identical_strings_score_at_one():
    r = await BleuScorer().score("q", "The capital of France is Paris.", "The capital of France is Paris.")
    assert r.passed is True
    assert r.score == pytest.approx(1.0, abs=1e-6)


async def test_unrelated_strings_score_near_zero():
    r = await BleuScorer().score(
        "q", "Water boils at 100 degrees Celsius.", "The capital of France is Paris."
    )
    assert r.passed is False
    assert r.score < 0.1


async def test_prose_wrapped_short_answer_scores_low():
    """The Finding 1 case: BLEU on a 1-word reference vs a full sentence
    is structurally near-zero. Brevity penalty inverts (output longer
    than reference) and 4-gram overlap with a 1-token reference is 0.

    This is BLEU's headline failure mode on short factual Q&A — the
    point of having BLEU in the suite is to demonstrate it on this
    corpus, not to tune around it.
    """
    r = await BleuScorer().score(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Paris",
    )
    assert r.passed is False
    assert r.score < 0.30


async def test_score_is_in_unit_interval():
    r = await BleuScorer().score("q", "alpha beta gamma", "delta epsilon zeta")
    assert 0.0 <= r.score <= 1.0


async def test_threshold_gates_verdict():
    output = "the cat sat on the mat"
    expected = "the cat sat on the mat"
    lenient = await BleuScorer(threshold=0.1).score("q", output, expected)
    strict = await BleuScorer(threshold=0.99).score("q", output, expected)
    assert lenient.passed is True
    assert lenient.score == pytest.approx(strict.score, abs=1e-6)


async def test_reason_mentions_threshold():
    r = await BleuScorer().score("q", "Paris", "Paris")
    assert "threshold" in r.reason
    assert "bleu" in r.reason


def test_threshold_validated_at_construction():
    with pytest.raises(ValueError):
        BleuScorer(threshold=1.5)
    with pytest.raises(ValueError):
        BleuScorer(threshold=-0.1)
