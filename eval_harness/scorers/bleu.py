from __future__ import annotations

from sacrebleu.metrics import BLEU

from eval_harness.scorers.base import Scorer, ScoreResult

DEFAULT_THRESHOLD = 0.30


class BleuScorer(Scorer):
    """How much vocabulary do the model's answer and the expected answer share?

    Explanation: BLEU counts how many short word-sequences
    appear in both texts. An *n-gram* is just a run of n consecutive
    words: "the cat" is a 2-gram, "the cat sat" is a 3-gram. BLEU
    checks 1-grams, 2-grams, 3-grams, and 4-grams, then combines the
    counts. Two corrections that matter:

    - *Brevity penalty* — if the model's answer is much shorter than
      the reference, BLEU multiplies the score down. Otherwise a model
      could just emit one well-chosen word and get high precision for
      free.
    - *Clipping* — if the model says "the the the" and "the" appears
      once in the reference, only one of those gets credit.

    BLEU does not understand meaning. It only counts word sequences. It
    was invented in 2002 for machine translation, where the reference
    is a full human-translated sentence. Score is `sacrebleu`'s 0-100
    scale rescaled to [0, 1].

    Default threshold 0.30 follows translation-paper convention
    ("decent translation"). On this project's short-factual corpus
    BLEU collapses near zero on prose-wrapped answers like exact match
    does, but lifts above threshold on prose-vs-prose pairs.
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
