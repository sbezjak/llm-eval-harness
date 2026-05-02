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
    """Pin BLEU's main failure mode for this kind of dataset.

    The model gives a perfectly correct answer ("The capital of France
    is Paris.") but the expected value is just "Paris". A human grader
    would mark this PASS. BLEU scores it near zero, because:

    - the expected reference has 1 word, so there are no 2-grams,
      3-grams, or 4-grams to match on at all;
    - the brevity penalty further pushes the score down because the
      output is much longer than the reference.

    The test asserts the verdict is FAIL on this pair, locking that
    behavior in. If BLEU ever started passing this case (because of a
    library change or a tokenizer change), we want to be told.
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
