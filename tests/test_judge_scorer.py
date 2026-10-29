from __future__ import annotations

import json

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
    raw = (
        '{"evidence_span": "the answer text", "contradicts_expected": false, '
        '"correctness": 9, "relevance": 10}'
    )
    c, r, span, contra = _parse_judge_response(raw)
    assert (c, r) == (9, 10)
    assert span == "the answer text"
    assert contra is False


def test_parser_extracts_json_from_surrounding_prose():
    """Models often wrap JSON in markdown fences or chatty text. The parser
    should find the JSON blob anyway."""
    raw = (
        "Sure! Here is my evaluation:\n"
        "```json\n"
        '{"evidence_span": "Paris", "contradicts_expected": false, '
        '"correctness": 8, "relevance": 9}\n'
        "```\n"
    )
    c, r, span, _ = _parse_judge_response(raw)
    assert (c, r) == (8, 9)
    assert span == "Paris"


def test_parser_reads_contradiction_flag():
    """The contradicts_expected flag must survive parsing - it gates the
    correctness score in score()."""
    raw = (
        '{"evidence_span": "the moon is made of cheese", '
        '"contradicts_expected": true, "correctness": 8, "relevance": 9}'
    )
    _, _, _, contra = _parse_judge_response(raw)
    assert contra is True


def test_parser_falls_back_to_regex_on_malformed_json():
    """If JSON parsing fails entirely, grab the integers by regex. No grounding
    fields survive that path - the empty span then correctly fails the substring
    check in score(), because an unparseable rationale is not a grounded one."""
    raw = "correctness: 7, relevance: 8 (the answer was decent)"
    c, r, span, contra = _parse_judge_response(raw)
    assert (c, r) == (7, 8)
    assert span == ""
    assert contra is False


def test_parser_raises_when_nothing_parseable():
    """Don't silently coerce to 0 - that would hide a broken judge."""
    with pytest.raises(JudgeParseError):
        _parse_judge_response("I don't know how to evaluate this, sorry.")


# ---------- scorer integration tests ----------


def _resp(correctness: int, relevance: int, span: str, contradicts: bool = False) -> str:
    """A well-formed judge response whose evidence_span the test controls."""
    return json.dumps(
        {
            "evidence_span": span,
            "contradicts_expected": contradicts,
            "correctness": correctness,
            "relevance": relevance,
        }
    )


@pytest.mark.mocked
async def test_judge_passes_when_both_dimensions_high():
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(_resp(10, 10, "answer")))
    r = await scorer.score("Q", "answer", "expected")
    assert r.passed is True
    assert r.score == 1.0
    assert "correctness=10/10" in r.reason


@pytest.mark.mocked
async def test_judge_fails_when_combined_score_below_threshold():
    """correctness=5, relevance=5 → score 0.5, below 0.7 → fail."""
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(_resp(5, 5, "answer")))
    r = await scorer.score("Q", "answer", "expected")
    assert r.passed is False
    assert r.score == 0.5


@pytest.mark.mocked
async def test_judge_clamps_out_of_range_scores():
    """Small models occasionally invent 11/10 scores. Clamp, don't crash."""
    scorer = LLMJudgeScorer(threshold=0.5, judge_fn=_fake_judge(_resp(15, -3, "answer")))
    r = await scorer.score("Q", "answer", "expected")
    # 10 + 0 (clamped) → 0.5
    assert r.score == 0.5


@pytest.mark.mocked
async def test_judge_threshold_gates_verdict():
    """Same response, two thresholds, opposite verdicts."""
    resp = _resp(7, 7, "answer")
    lenient = LLMJudgeScorer(threshold=0.5, judge_fn=_fake_judge(resp))
    strict = LLMJudgeScorer(threshold=0.9, judge_fn=_fake_judge(resp))

    assert (await lenient.score("Q", "answer", "E")).passed is True
    assert (await strict.score("Q", "answer", "E")).passed is False


# ---------- grounding guard (the anti-lie substring check) ----------


@pytest.mark.mocked
async def test_judge_rejects_fabricated_evidence_span():
    """The core guard: a judge that cites a span NOT in the answer is
    fabricating its rationale. The substring check catches it in code - the
    judge cannot claim to have read what it cannot quote."""
    resp = _resp(10, 10, "a span that never appears in the answer")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    with pytest.raises(JudgeParseError, match="fabricated"):
        await scorer.score("Q", "the real answer text", "E")


@pytest.mark.mocked
async def test_judge_rejects_ungrounded_verdict():
    """An empty evidence_span is an un-grounded verdict - the 'I read it' lie
    with no quote at all. Not trusted."""
    resp = _resp(10, 10, "")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    with pytest.raises(JudgeParseError, match="un-grounded"):
        await scorer.score("Q", "the real answer text", "E")


@pytest.mark.mocked
async def test_judge_accepts_real_evidence_span():
    """A verbatim quote that really is in the answer passes the guard."""
    resp = _resp(9, 9, "capital of France is Paris")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    r = await scorer.score("Q", "The capital of France is Paris.", "Paris")
    assert r.passed is True
    assert "grounded on:" in r.reason


@pytest.mark.mocked
async def test_guard_tolerates_whitespace_reflow():
    """A weak judge re-flows whitespace when quoting (observed live: it
    collapsed a double newline). That is cosmetic, not a lie - the guard must
    still accept it."""
    # answer has a double newline; the judge quotes it with a single one
    answer = "There are eight planets:\n\n1. Mercury\n2. Venus"
    resp = _resp(10, 10, "There are eight planets:\n1. Mercury\n2. Venus")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    r = await scorer.score("Q", answer, "8")
    assert r.passed is True


@pytest.mark.mocked
async def test_guard_tolerates_smart_quote_substitution():
    """A weak judge swaps straight quotes for curly ones when quoting
    (observed live: it quoted the answer's "AI" as ’AI’). Cosmetic - accept."""
    answer = 'The acronym "AI" stands for Artificial Intelligence.'
    resp = _resp(10, 10, "The acronym ’AI’ stands for Artificial Intelligence.")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    r = await scorer.score("Q", answer, "Artificial Intelligence")
    assert r.passed is True


@pytest.mark.mocked
async def test_guard_still_catches_paraphrase_after_normalization():
    """Normalization forgives quotes and whitespace, NOT different words. A
    span with invented tokens (observed live: 'pysto', options never in the
    answer) still fails - the guard keeps its teeth."""
    answer = "Use pytest -m 'slow or network' to run those tests."
    resp = _resp(10, 10, "run pysto --junit-xml-filter='slow|network'")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    with pytest.raises(JudgeParseError, match="fabricated"):
        await scorer.score("Q", answer, "pytest -m 'slow or network'")


@pytest.mark.mocked
async def test_contradiction_flag_caps_correctness():
    """A judge that admits the answer contradicts the reference cannot also
    pass it on correctness - correctness is capped at 2 regardless of the
    number the judge wrote."""
    resp = _resp(10, 10, "the moon is made of cheese", contradicts=True)
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    r = await scorer.score("Q", "the moon is made of cheese", "The moon is rock.")
    # correctness capped 10 → 2, relevance 10 → (2+10)/20 = 0.6, below 0.7
    assert r.score == 0.6
    assert r.passed is False
    assert "contradiction with EXPECTED" in r.reason


@pytest.mark.mocked
async def test_judge_prompt_contains_all_three_inputs():
    """The rubric is useless if the judge can't see the question, the
    expected answer, and the model's output. Pin that the template
    actually substitutes all three."""
    captured: list[str] = []
    resp = _resp(8, 8, "Paris")
    scorer = LLMJudgeScorer(judge_fn=_capturing_judge(resp, captured))
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
async def test_judge_reason_includes_dimension_breakdown_and_grounding():
    """The `reason` string is what shows up in the pytest report when a test
    fails - for a QA engineer scanning failures, this is the main evidence.
    It must surface the per-dimension scores (a right-but-off-topic answer and
    a confident-hallucination answer are distinguishable only per-dimension)
    AND the verbatim span the judge grounded its verdict on."""
    resp = _resp(2, 9, "on topic but wrong")
    scorer = LLMJudgeScorer(threshold=0.7, judge_fn=_fake_judge(resp))
    r = await scorer.score("Q", "on topic but wrong", "E")
    assert "correctness=2/10" in r.reason
    assert "relevance=9/10" in r.reason
    assert "grounded on:" in r.reason
    assert "on topic but wrong" in r.reason
