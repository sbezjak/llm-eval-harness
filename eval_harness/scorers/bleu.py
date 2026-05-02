from __future__ import annotations

from sacrebleu.metrics import BLEU

from eval_harness.scorers.base import Scorer, ScoreResult

DEFAULT_THRESHOLD = 0.30


class BleuScorer(Scorer):
    """Sentence-level BLEU between `output` (hypothesis) and `expected` (reference).

    BLEU was designed for machine translation: how much of the model's
    n-gram vocabulary appears in a human reference translation? It is an
    *n-gram overlap* metric — it does not understand meaning, only counts
    word sequences (with brevity penalty and clipping).

    Backed by `sacrebleu` (the production-standard implementation; what
    translation papers report) rather than `nltk.translate.bleu_score`,
    for reproducible tokenization across runs and machines. Score is
    sacrebleu's 0-100 scale rescaled to [0, 1].

    Default threshold 0.30 follows translation-paper convention
    ("decent translation"). For short factual Q&A this threshold is
    expected to fail almost every prose-wrapped right answer — the same
    structural failure mode as exact match (Finding 1). That is the
    finding this scorer is here to demonstrate, not a bug to tune around.
    """

    name = "bleu"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self._bleu = BLEU(effective_order=True)

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        result = self._bleu.sentence_score(output, [expected])
        normalized = result.score / 100.0
        passed = normalized >= self.threshold
        return ScoreResult(
            passed=passed,
            score=normalized,
            reason=f"bleu={normalized:.3f} {'>=' if passed else '<'} threshold={self.threshold}",
        )
