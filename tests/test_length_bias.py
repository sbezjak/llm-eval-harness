"""Does the judge score longer answers higher than shorter ones,
even when both answers are factually correct?

This is a known failure mode for LLM judges: many of them learned from
human preference data where longer answers were rated more thorough,
and they pick up the bias. A judge that does this will rubber-stamp
verbose answers and unfairly fail concise ones.

How the test works:

- Three cases (capital of France, chemical symbol of silver, why ice
  floats), each with a question and *two* correct answers — one short
  ("Paris.") and one long (a paragraph that elaborates).
- For each case, ask the judge to score both versions of the same
  correct answer. Both should pass (they're both correct), and the
  test asserts that.
- The interesting output is the delta between the long and short
  scores, printed for each case. A systematic positive delta would
  point at length bias; near-zero deltas are evidence the rubric is
  not biased on this kind of short-factual question.

Caveat. n=3 is far too small to claim "this judge has no length bias
in general." It's enough to claim "the judge gave both versions the
same score on these three cases." Real length-bias work uses dozens
to hundreds of paired prompts across topics and lengths.

A useful null result is still a result: if the test prints zero
deltas, that's evidence the rubric design (separate correctness +
relevance scores, reasoning before score) probably suppresses length
bias on this kind of question. If a future model or rubric change
makes deltas appear, you'll see them here.
"""

from __future__ import annotations

import pytest

from eval_harness.scorers.judge import LLMJudgeScorer

# Each case: (id, question, expected, short_correct, long_correct).
# Both short_correct and long_correct are factually correct answers to
# the question. They differ only in length and elaboration.
CASES = [
    (
        "capital_france",
        "What is the capital of France?",
        "Paris",
        "Paris.",
        (
            "Paris is the capital and largest city of France, located in the "
            "north-central part of the country along the Seine River. The "
            "city has been the capital of France since 987 CE under Hugh "
            "Capet, and today it serves as the political, cultural, and "
            "economic center of the nation, with a metropolitan population "
            "of over 12 million people. It is also one of the most visited "
            "cities in the world."
        ),
    ),
    (
        "silver_symbol",
        "What is the chemical symbol for silver?",
        "Ag",
        "Ag.",
        (
            "The chemical symbol for silver is Ag, derived from the Latin "
            "word 'argentum' meaning silver. Silver has atomic number 47 on "
            "the periodic table and is classified as a transition metal. "
            "Like other elements, its symbol is used in chemical equations "
            "and on the periodic table to represent the element concisely. "
            "The symbol Ag has been in use since the medieval period when "
            "Latin was the language of science."
        ),
    ),
    (
        "ice_floats",
        "Why does ice float on water?",
        "Ice is less dense than liquid water.",
        "Because ice is less dense than liquid water.",
        (
            "Ice floats on water because it is less dense than liquid water. "
            "When water freezes, the molecules arrange themselves into a "
            "crystalline lattice structure held together by hydrogen bonds, "
            "and this structure takes up more space than the same number of "
            "water molecules in liquid form. Because density is mass divided "
            "by volume, the same mass occupying a larger volume gives ice a "
            "lower density than liquid water (about 0.92 g/cm³ for ice vs "
            "1.00 g/cm³ for water at 4°C), so it floats."
        ),
    ),
]


@pytest.mark.ollama
@pytest.mark.parametrize(
    "case_id,question,expected,short_output,long_output",
    CASES,
    ids=[c[0] for c in CASES],
)
async def test_judge_length_bias(
    case_id: str,
    question: str,
    expected: str,
    short_output: str,
    long_output: str,
):
    judge = LLMJudgeScorer(threshold=0.7)

    short_result = await judge.score(question, short_output, expected)
    long_result = await judge.score(question, long_output, expected)
    delta = long_result.score - short_result.score

    print(f"\ncase:         {case_id}")
    print(f"question:     {question}")
    print(f"expected:     {expected!r}")
    print(
        f"short ({len(short_output):>4} chars): score={short_result.score:.3f} "
        f"passed={short_result.passed}"
    )
    print(
        f"long  ({len(long_output):>4} chars): score={long_result.score:.3f} "
        f"passed={long_result.passed}"
    )
    print(f"delta (long - short): {delta:+.3f}")
    if delta > 0.05:
        print("  → long version scored meaningfully higher — possible length bias")
    elif delta < -0.05:
        print("  → short version scored higher — opposite of length bias")
    else:
        print("  → no meaningful difference on this case")

    assert short_result.passed, (
        f"\n  judge wrongly failed the SHORT correct answer for {case_id!r}:"
        f"\n  score={short_result.score:.3f} reason={short_result.reason}"
    )
    assert long_result.passed, (
        f"\n  judge wrongly failed the LONG correct answer for {case_id!r}:"
        f"\n  score={long_result.score:.3f} reason={long_result.reason}"
    )
