from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

from eval_harness.scorers.base import Scorer, ScoreResult

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.75


class SemanticScorer(Scorer):
    """Cosine similarity between embeddings of `output` and `expected`.

    Embeddings come from a sentence-transformers model (default
    `all-MiniLM-L6-v2`: small, fast, good enough for short factual answers).
    The cosine score is mapped from [-1, 1] to [0, 1] by clamping negatives
    to 0; for natural English text the raw value is almost always positive,
    but the contract on `ScoreResult.score` is [0, 1] and we honor it.

    `passed` is `score >= threshold`. The threshold is the knob you tune
    against your own dataset — too high collapses into exact match, too low
    lets unrelated answers through. 0.75 is a sensible starting point for
    `all-MiniLM-L6-v2` on short answers; expect to tune.
    """

    name = "semantic"

    # Class-level cache: loading a model takes seconds and downloads weights
    # on first use. One process, one model load per model name, regardless
    # of how many SemanticScorer instances are created.
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.model_name = model_name
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        self._model = self._model_cache[model_name]

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        embeddings = self._model.encode([output, expected], convert_to_tensor=True)
        cosine = util.cos_sim(embeddings[0], embeddings[1]).item()
        clamped = max(0.0, min(1.0, cosine))
        passed = clamped >= self.threshold
        return ScoreResult(
            passed=passed,
            score=clamped,
            reason=f"cosine={clamped:.3f} {'>=' if passed else '<'} threshold={self.threshold}",
        )
