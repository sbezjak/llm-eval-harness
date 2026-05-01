from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of a single scorer applied to a single (output, expected) pair.

    `score` is in [0.0, 1.0]. `passed` is the scorer's binary verdict — for
    threshold-based scorers (semantic, judge), it's `score >= threshold`.
    `reason` is a short human-readable explanation, surfaced in reports.
    """

    passed: bool
    score: float
    reason: str


class Scorer(ABC):
    """Pure scoring function over (question, output, expected).

    Scorers must not perform I/O at score time — that keeps unit tests fast
    and deterministic. Backends that need I/O (e.g. embedding models, judge
    LLMs) load their resources at construction time.
    """

    name: str

    @abstractmethod
    def score(self, question: str, output: str, expected: str) -> ScoreResult: ...
