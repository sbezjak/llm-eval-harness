"""Calibration: do the scorers agree with human verdicts on real model output?

For each (item, scorer) pair, run the scorer against a frozen Ollama
output from `data/human_labels.yaml` and assert the scorer's PASS/FAIL
matches the human verdict.

Items where the scorer is *known* to disagree with humans for a
structural reason (exact match doesn't accept prose; semantic embeddings
fail on shape mismatch; the LLM judge passes its own hallucinations) are
marked `xfail(strict=True)` with a plain-English `reason=` describing
exactly what the scorer does wrong on that item. The xfail set IS the
contract: an unexpected pass means a documented limitation has gone
away — break the suite so the limitation isn't silently lost.

The corpus is frozen (committed outputs, not re-generated each run) so
this test measures the scorers, not the model. Re-running the model is
a separate concern handled by `test_eval_pipeline.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval_harness.scorers.base import ScoreResult, Scorer
from eval_harness.scorers.bleu import BleuScorer
from eval_harness.scorers.exact_match import ExactMatchScorer
from eval_harness.scorers.judge import LLMJudgeScorer
from eval_harness.scorers.rouge import RougeScorer
from eval_harness.scorers.semantic import SemanticScorer
from tests._xfail import with_xfail

HUMAN_LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "human_labels.yaml"


def _load_corpus() -> list[dict]:
    with HUMAN_LABELS_PATH.open() as f:
        return yaml.safe_load(f)


CORPUS = _load_corpus()
CORPUS_BY_ID = {item["id"]: item for item in CORPUS}

# Each map: item_id -> plain-English reason describing what the scorer
# does wrong on that item. The reason surfaces in pytest output, so a
# reader can understand the failure without reading any other file.

_EXACT_PROSE = (
    "exact match wrongly says FAIL: model returns the right answer wrapped "
    "in conversational prose, so output != expected"
)
EXACT_MATCH_KNOWN_DISAGREEMENTS = {
    "factual_001": _EXACT_PROSE,
    "factual_002": _EXACT_PROSE,
    "factual_003": _EXACT_PROSE,
    "factual_004": _EXACT_PROSE,
    "definition_001": _EXACT_PROSE,
    "definition_002": _EXACT_PROSE,
    "reasoning_001": _EXACT_PROSE,
    "reasoning_002": _EXACT_PROSE,
    "reasoning_003": _EXACT_PROSE,
    # procedural_001 is intentionally absent: exact match correctly says
    # FAIL on the hallucinated output and the human also says FAIL. They
    # agree.
}

SEMANTIC_KNOWN_DISAGREEMENTS = {
    "factual_002": (
        "semantic wrongly says FAIL: expected is a 2-character symbol ('Ag'); "
        "embedding distance to a full sentence is too large"
    ),
    "factual_003": (
        "semantic wrongly says FAIL: short expected ('South America') vs "
        "prose answer pulls cosine below 0.75"
    ),
    "factual_004": (
        "semantic wrongly says FAIL: model returns a bulleted list of all 8 "
        "planets; expected is the digit '8' — embedding can't see past the "
        "shape mismatch"
    ),
    "definition_001": (
        "semantic wrongly says FAIL: short expected ('Artificial Intelligence') "
        "vs full-sentence answer"
    ),
    "definition_002": (
        "semantic wrongly says FAIL: same idea phrased with different words; "
        "embedding similarity below threshold"
    ),
}

# BLEU and ROUGE agree with humans when the expected reference is a
# full prose sentence with vocabulary that overlaps the model's answer.
# They disagree (say FAIL when the human says PASS) when the expected
# reference is a short bare token like "Paris", "Ag", or "8" — there
# aren't enough words to compute meaningful overlap against, so the
# scores collapse near zero the same way exact match's verdict does.

_BLEU_SHORT_REF = (
    "bleu wrongly says FAIL: expected reference is short (≤25 chars), so "
    "n-gram overlap with the prose-wrapped output is structurally near "
    "zero — same failure mode as exact match on this corpus"
)
BLEU_KNOWN_DISAGREEMENTS = {
    "factual_001": _BLEU_SHORT_REF,    # expected: "Ljubljana"
    "factual_002": _BLEU_SHORT_REF,    # expected: "Ag"
    "factual_003": _BLEU_SHORT_REF,    # expected: "South America"
    "factual_004": _BLEU_SHORT_REF,    # expected: "8"
    "definition_001": _BLEU_SHORT_REF, # expected: "Artificial Intelligence"
    "reasoning_001": _BLEU_SHORT_REF,  # short expected vs long-form output
    "reasoning_003": _BLEU_SHORT_REF,  # mid-length expected vs 2000+ char output
    # definition_002, reasoning_002: expected is a full prose sentence
    # whose vocabulary overlaps the output enough that BLEU clears 0.30
    # and agrees with the human. These are agreements, not disagreements.
    # procedural_001: BLEU says FAIL, human says FAIL — they agree by
    # accident (the hallucinated output happens to share little
    # vocabulary with the expected text).
}

_ROUGE_SHORT_REF = (
    "rouge wrongly says FAIL: expected reference is short, so ROUGE-L "
    "precision is 1/N over the long output and F1 falls below threshold "
    "even when recall is high"
)
ROUGE_KNOWN_DISAGREEMENTS = {
    "factual_001": _ROUGE_SHORT_REF,
    "factual_002": _ROUGE_SHORT_REF,
    "factual_003": _ROUGE_SHORT_REF,
    "factual_004": _ROUGE_SHORT_REF,
    "reasoning_001": _ROUGE_SHORT_REF,
    "reasoning_003": _ROUGE_SHORT_REF,
    # definition_001 ("Artificial Intelligence") is short but ROUGE-L
    # squeaks above 0.40 because both the reference and the output
    # contain the full phrase verbatim — recall pulls F1 just over the
    # line. BLEU still fails it (4-gram overlap is too sparse).
    # definition_002, reasoning_002: prose-vs-prose with high vocabulary
    # overlap; ROUGE clears 0.40 cleanly.
    # procedural_001: ROUGE says FAIL, human says FAIL — agree by accident.
}

JUDGE_KNOWN_DISAGREEMENTS = {
    "procedural_001": (
        "judge wrongly says PASS: the model's answer is a fully hallucinated "
        "pytest flag ('--junit-xml-filter') that does not exist; the same "
        "llama3.2 acting as judge can't detect its own hallucination "
        "(this is self-grading bias — the canonical LLM-as-judge failure mode)"
    ),
}


def _params_with_xfail(known_disagreements: dict[str, str]):
    return with_xfail([item["id"] for item in CORPUS], known_disagreements)


async def _agreement_assertion(scorer: Scorer, item_id: str) -> ScoreResult:
    item = CORPUS_BY_ID[item_id]
    result = await scorer.score(item["question"], item["frozen_output"], item["expected"])
    expected_pass = item["human_verdict"] == "PASS"

    print(f"\nitem:           {item['id']}")
    print(f"question:       {item['question']}")
    print(f"expected:       {item['expected']!r}")
    print(f"frozen_output:  {item['frozen_output']!r}")
    print(f"human_verdict:  {item['human_verdict']}")
    print(f"scorer:         {scorer.name}")
    print(f"scorer score:   {result.score:.3f}")
    print(f"scorer passed:  {result.passed}")
    print(f"scorer reason:  {result.reason}")

    assert result.passed == expected_pass, (
        f"\n  {scorer.name} disagrees with human on {item_id!r}:"
        f"\n    human:  {item['human_verdict']}"
        f"\n    scorer: {'PASS' if result.passed else 'FAIL'} (score={result.score:.3f})"
        f"\n    reason: {result.reason}"
    )
    return result


@pytest.mark.mocked
@pytest.mark.parametrize(
    "item_id",
    _params_with_xfail(EXACT_MATCH_KNOWN_DISAGREEMENTS),
)
async def test_exact_match_calibration(item_id: str):
    await _agreement_assertion(ExactMatchScorer(), item_id)


@pytest.mark.mocked
@pytest.mark.parametrize(
    "item_id",
    _params_with_xfail(SEMANTIC_KNOWN_DISAGREEMENTS),
)
async def test_semantic_calibration(item_id: str):
    await _agreement_assertion(SemanticScorer(threshold=0.75), item_id)


@pytest.mark.mocked
@pytest.mark.parametrize(
    "item_id",
    _params_with_xfail(BLEU_KNOWN_DISAGREEMENTS),
)
async def test_bleu_calibration(item_id: str):
    await _agreement_assertion(BleuScorer(), item_id)


@pytest.mark.mocked
@pytest.mark.parametrize(
    "item_id",
    _params_with_xfail(ROUGE_KNOWN_DISAGREEMENTS),
)
async def test_rouge_calibration(item_id: str):
    await _agreement_assertion(RougeScorer(), item_id)


@pytest.mark.ollama
@pytest.mark.parametrize(
    "item_id",
    _params_with_xfail(JUDGE_KNOWN_DISAGREEMENTS),
)
async def test_judge_calibration(item_id: str):
    """Judge calibration runs against live Ollama.

    The frozen output is fixed, but the judge's score on that output is
    NOT — llama3.2's judge call has run-to-run variance even at temp=0.
    We accept that variance is part of what we're measuring: if the judge
    flips PASS/FAIL on a non-xfailed item between runs, that itself is a
    finding (judge instability is a real phenomenon and worth surfacing).

    Marked @ollama and excluded from the fast test path for that reason.
    """
    await _agreement_assertion(LLMJudgeScorer(threshold=0.7), item_id)
