from __future__ import annotations

from rouge_score import rouge_scorer

from eval_harness.scorers.base import Scorer, ScoreResult

DEFAULT_THRESHOLD = 0.40
DEFAULT_VARIANT = "rougeL"


class RougeScorer(Scorer):
    """ROUGE-L F1 between `output` (hypothesis) and `expected` (reference).

    ROUGE was designed for summarization: did the model's summary cover
    the content of a human-written reference summary? Where BLEU
    emphasizes precision, ROUGE emphasizes recall. ROUGE-L uses the
    *longest common subsequence* (matches don't have to be contiguous,
    just same order), which makes it more forgiving than BLEU on short
    answers — but still vocabulary-bound.

    Backed by `rouge-score` (Google's reference implementation, used by
    summarization papers). Default variant is ROUGE-L F1; default
    threshold 0.40 follows summarization-paper convention.

    Like BLEU, this scorer is expected to fail on the prose-wrapped
    short-factual cases on this corpus — n-gram overlap is structurally
    the wrong question for "Paris" vs "The capital of France is Paris."
    """

    name = "rouge"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.variant = variant
        self._scorer = rouge_scorer.RougeScorer([variant], use_stemmer=True)

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        scores = self._scorer.score(expected, output)
        f1 = scores[self.variant].fmeasure
        passed = f1 >= self.threshold
        return ScoreResult(
            passed=passed,
            score=f1,
            reason=(
                f"{self.variant}_f1={f1:.3f} "
                f"{'>=' if passed else '<'} threshold={self.threshold}"
            ),
        )
