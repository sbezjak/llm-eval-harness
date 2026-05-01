from __future__ import annotations

from eval_harness.scorers.exact_match import ExactMatchScorer


async def test_passes_on_identical_strings():
    r = await ExactMatchScorer().score("q", "Paris", "Paris")
    assert r.passed is True
    assert r.score == 1.0


async def test_fails_when_model_adds_prose():
    r = await ExactMatchScorer().score("q", "The capital of France is Paris.", "Paris")
    assert r.passed is False
    assert r.score == 0.0


async def test_fails_on_case_difference():
    """Strict by design — no normalization, no lowercasing."""
    r = await ExactMatchScorer().score("q", "paris", "Paris")
    assert r.passed is False


async def test_fails_on_trailing_whitespace():
    """Strict by design — no whitespace collapsing."""
    r = await ExactMatchScorer().score("q", "Paris\n", "Paris")
    assert r.passed is False


async def test_reason_is_populated():
    r_pass = await ExactMatchScorer().score("q", "x", "x")
    r_fail = await ExactMatchScorer().score("q", "x", "y")
    assert r_pass.reason
    assert r_fail.reason
    assert r_pass.reason != r_fail.reason
