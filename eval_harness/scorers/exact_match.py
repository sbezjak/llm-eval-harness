from __future__ import annotations

from eval_harness.scorers.base import Scorer, ScoreResult


class ExactMatchScorer(Scorer):
    """Strict character-for-character match.

    No normalization (no lowercasing, no whitespace collapsing, no substring
    fallback). On open-ended LLM output this scorer fails almost always —
    that's the point. It is included as the honest baseline that motivates
    semantic similarity and LLM-as-judge.
    """

    name = "exact_match"

    def score(self, question: str, output: str, expected: str) -> ScoreResult:
        passed = output == expected
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="exact match" if passed else "outputs differ",
        )
