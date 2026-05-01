from __future__ import annotations

import pytest

from eval_harness.scorers.judge import (
    JudgeParseError,
    LLMJudgeScorer,
    _parse_judge_response,
)


def _fake_judge(response: str):
    """Build a judge_fn that returns a fixed string, ignoring the prompt."""

    async def _fn(prompt: str) -> str:
        return response

    return _fn


def _capturing_judge(response: str, captured: list[str]):
    """Like _fake_judge but records the prompt it was called with."""

    async def _fn(prompt: str) -> str:
        captured.append(prompt)
        return response

    return _fn


# ---------- parser unit tests ----------


def test_parser_accepts_clean_json():
    raw = '{"reasoning": "looks right", "correctness": 9, "relevance": 10}'
    c, r, reasoning = _parse_judge_response(raw)
    assert (c, r) == (9, 10)
    assert reasoning == "looks right"


def test_parser_extracts_json_from_surrounding_prose():
    """Models often wrap JSON in markdown fences or chatty text. The parser
    should find the JSON blob anyway."""
    raw = (
        "Sure! Here is my evaluation:\n"
        "```json\n"
        '{"reasoning": "correct", "correctness": 8, "relevance": 9}\n'
        "```\n"
    )
    c, r, _ = _parse_judge_response(raw)
    assert (c, r) == (8, 9)


def test_parser_falls_back_to_regex_on_malformed_json():
    """If JSON parsing fails entirely, grab the integers by regex. This is
    the safety net for when a small local model produces almost-but-not-quite
    JSON (trailing commas, smart quotes, missing braces)."""
    raw = "correctness: 7, relevance: 8 (the answer was decent)"
    c, r, reasoning = _parse_judge_response(raw)
    assert (c, r) == (7, 8)
    assert "regex fallback" in reasoning


def test_parser_raises_when_nothing_parseable():
    """Don't silently coerce to 0 — that would hide a broken judge."""
    with pytest.raises(JudgeParseError):
        _parse_judge_response("I don't know how to evaluate this, sorry.")


# ---------- scorer integration tests ----------


@pytest.mark.mocked
async def test_judge_passes_when_both_dimensions_high():
    raw = '{"reasoning": "perfect", "correctness": 10, "relevance": 10}'
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(raw))
    r = await scorer.score("Q", "A", "expected")
    assert r.passed is True
    assert r.score == 1.0
    assert "correctness=10/10" in r.reason


@pytest.mark.mocked
async def test_judge_fails_when_combined_score_below_threshold():
    """correctness=5, relevance=5 → score 0.5, below 0.7 → fail."""
    raw = '{"reasoning": "meh", "correctness": 5, "relevance": 5}'
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(raw))
    r = await scorer.score("Q", "A", "expected")
    assert r.passed is False
    assert r.score == 0.5


@pytest.mark.mocked
async def test_judge_clamps_out_of_range_scores():
    """Small models occasionally invent 11/10 scores. Clamp, don't crash."""
    raw = '{"reasoning": "great", "correctness": 15, "relevance": -3}'
    scorer = LLMJudgeScorer(threshold=0.5, judge_fn=_fake_judge(raw))
    r = await scorer.score("Q", "A", "expected")
    # 10 + 0 (clamped) → 0.5
    assert r.score == 0.5


@pytest.mark.mocked
async def test_judge_threshold_gates_verdict():
    """Same response, two thresholds, opposite verdicts."""
    raw = '{"reasoning": "ok", "correctness": 7, "relevance": 7}'
    lenient = LLMJudgeScorer(threshold=0.5, judge_fn=_fake_judge(raw))
    strict = LLMJudgeScorer(threshold=0.9, judge_fn=_fake_judge(raw))

    assert (await lenient.score("Q", "A", "E")).passed is True
    assert (await strict.score("Q", "A", "E")).passed is False


@pytest.mark.mocked
async def test_judge_prompt_contains_all_three_inputs():
    """The rubric is useless if the judge can't see the question, the
    expected answer, and the model's output. Pin that the template
    actually substitutes all three."""
    captured: list[str] = []
    raw = '{"reasoning": "ok", "correctness": 8, "relevance": 8}'
    scorer = LLMJudgeScorer(judge_fn=_capturing_judge(raw, captured))
    await scorer.score(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Paris",
    )
    assert len(captured) == 1
    prompt = captured[0]
    assert "What is the capital of France?" in prompt
    assert "Paris" in prompt
    assert "The capital of France is Paris." in prompt


def test_build_prompt_is_inspectable_without_scoring():
    """`build_prompt` is public so debug paths can inspect the rendered
    rubric without spending a judge call. Confirms the same substitution
    used by score()."""
    scorer = LLMJudgeScorer()
    prompt = scorer.build_prompt(
        "Q?",
        "the model said this",
        "the expected was that",
    )
    assert "Q?" in prompt
    assert "the model said this" in prompt
    assert "the expected was that" in prompt


@pytest.mark.mocked
async def test_judge_propagates_parse_error():
    """A garbage judge response should fail loud, not silently score 0."""
    scorer = LLMJudgeScorer(judge_fn=_fake_judge("???"))
    with pytest.raises(JudgeParseError):
        await scorer.score("Q", "A", "E")


def test_judge_threshold_validated_at_construction():
    with pytest.raises(ValueError):
        LLMJudgeScorer(threshold=1.5)
    with pytest.raises(ValueError):
        LLMJudgeScorer(threshold=-0.1)


@pytest.mark.mocked
async def test_judge_reason_includes_dimension_breakdown():
    """The disagreement report relies on per-scorer `reason` to explain
    why a scorer voted the way it did. For the judge specifically, the
    breakdown by dimension is the interesting signal."""
    raw = '{"reasoning": "wrong but on topic", "correctness": 2, "relevance": 9}'
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(raw))
    r = await scorer.score("Q", "A", "E")
    assert "correctness=2/10" in r.reason
    assert "relevance=9/10" in r.reason
    assert "wrong but on topic" in r.reason
