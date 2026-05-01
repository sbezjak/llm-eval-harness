from __future__ import annotations

from eval_harness.scorers.exact_match import ExactMatchScorer


def test_passes_on_identical_strings():
    r = ExactMatchScorer().score("q", "Paris", "Paris")
    assert r.passed is True
    assert r.score == 1.0


def test_fails_when_model_adds_prose():
    r = ExactMatchScorer().score("q", "The capital of France is Paris.", "Paris")
    assert r.passed is False
    assert r.score == 0.0


def test_fails_on_case_difference():
    """Strict by design — no normalization, no lowercasing."""
    r = ExactMatchScorer().score("q", "paris", "Paris")
    assert r.passed is False


def test_fails_on_trailing_whitespace():
    """Strict by design — no whitespace collapsing."""
    r = ExactMatchScorer().score("q", "Paris\n", "Paris")
    assert r.passed is False


def test_reason_is_populated():
    r_pass = ExactMatchScorer().score("q", "x", "x")
    r_fail = ExactMatchScorer().score("q", "x", "y")
    assert r_pass.reason
    assert r_fail.reason
    assert r_pass.reason != r_fail.reason
