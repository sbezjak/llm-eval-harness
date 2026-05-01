from __future__ import annotations

import pytest

from eval_harness.scorers.semantic import SemanticScorer


@pytest.fixture(scope="module")
def scorer() -> SemanticScorer:
    """One scorer for the whole module — model loads once."""
    return SemanticScorer(threshold=0.75)


def test_identical_strings_score_near_one(scorer: SemanticScorer):
    r = scorer.score("q", "Paris", "Paris")
    assert r.passed is True
    assert r.score == pytest.approx(1.0, abs=1e-3)


def test_paraphrase_scores_higher_than_unrelated(scorer: SemanticScorer):
    """The whole reason this scorer exists: prose-wrapped right answers
    sit much closer to the reference than unrelated text does.

    Note we do NOT assert that the paraphrase passes the default 0.75
    threshold. Empirically `all-MiniLM-L6-v2` puts "The capital of France
    is Paris." vs "Paris" around ~0.72 — a useful signal that 0.75 may be
    too strict for short bare references, and a tuning decision we want
    driven by the real golden set, not by this unit test.
    """
    paraphrase = scorer.score(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Paris",
    )
    unrelated = scorer.score(
        "What is the capital of France?",
        "Water boils at 100 degrees Celsius.",
        "Paris",
    )
    assert paraphrase.score > unrelated.score + 0.3
    assert paraphrase.score > 0.6


def test_unrelated_strings_score_low(scorer: SemanticScorer):
    r = scorer.score(
        "What is the capital of France?",
        "Water boils at 100 degrees Celsius.",
        "Paris",
    )
    assert r.passed is False
    assert r.score < 0.5


def test_threshold_gates_verdict():
    """Same pair, two thresholds, opposite verdicts."""
    output = "The capital of France is Paris."
    expected = "Paris"

    lenient = SemanticScorer(threshold=0.3).score("q", output, expected)
    strict = SemanticScorer(threshold=0.99).score("q", output, expected)

    assert lenient.passed is True
    assert strict.passed is False
    # Same pair, same score — only the verdict changes.
    assert lenient.score == pytest.approx(strict.score, abs=1e-6)


def test_score_is_in_unit_interval(scorer: SemanticScorer):
    r = scorer.score("q", "anything at all", "something else entirely")
    assert 0.0 <= r.score <= 1.0


def test_reason_mentions_threshold(scorer: SemanticScorer):
    r = scorer.score("q", "Paris", "Paris")
    assert "threshold" in r.reason
    assert "cosine" in r.reason


def test_threshold_validated_at_construction():
    with pytest.raises(ValueError):
        SemanticScorer(threshold=1.5)
    with pytest.raises(ValueError):
        SemanticScorer(threshold=-0.1)
