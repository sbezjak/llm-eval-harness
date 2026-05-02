"""Judge stability: run the judge N times on a frozen output, assert the
mean score stays within ±0.05 of an expected value.

The hypothesis going in was that llama3.2 at temp=0 would be slightly
noisy, with threshold-edge items (procedural_001 at 0.700) flipping
PASS/FAIL run-to-run. We observed the opposite: 5 runs returned 5
identical 0.700s. The judge isn't noisy here — it is *stuck*, passing
a hallucinated answer with the same exact score every time.

That's a worse production failure mode than flakiness:
  - Flaky judge → intermittent CI red → someone investigates.
  - Stuck judge → silent green every run → no one notices the wrong
    answer slipping through.

Averaging more runs doesn't fix a stuck judge — there's no noise to
average out. The fix is a different judge model.

This test locks the observation in place. If the mean ever drifts
outside ±0.05, the suite breaks — re-investigate with new data.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest
import yaml

from eval_harness.scorers.judge import LLMJudgeScorer

HUMAN_LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "human_labels.yaml"
N_RUNS = 5


def _load_corpus() -> list[dict]:
    with HUMAN_LABELS_PATH.open() as f:
        return yaml.safe_load(f)


CORPUS_BY_ID = {item["id"]: item for item in _load_corpus()}


@pytest.mark.ollama
@pytest.mark.parametrize(
    "item_id,expected_score",
    [
        # Stuck-at-threshold: hallucinated answer, judge consistently scores 0.700.
        ("procedural_001", 0.700),
        # Clearly correct, judge consistently scores 1.000 (sanity contrast).
        ("factual_001", 1.000),
    ],
)
async def test_judge_stability(item_id: str, expected_score: float):
    item = CORPUS_BY_ID[item_id]
    judge = LLMJudgeScorer(threshold=0.7)

    scores: list[float] = []
    verdicts: list[bool] = []
    for run in range(N_RUNS):
        result = await judge.score(item["question"], item["frozen_output"], item["expected"])
        scores.append(result.score)
        verdicts.append(result.passed)
        print(f"  run {run + 1}: score={result.score:.3f} passed={result.passed}")

    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(set(scores)) > 1 else 0.0
    flips = len(set(verdicts)) > 1

    print(f"\nitem:           {item_id}")
    print(f"runs:           {N_RUNS}")
    print(f"scores:         {[round(s, 3) for s in scores]}")
    print(f"mean:           {mean:.3f}")
    print(f"stdev:          {stdev:.3f}")
    print(f"min / max:      {min(scores):.3f} / {max(scores):.3f}")
    print(f"verdict flip:   {'YES — PASS/FAIL changed across runs' if flips else 'no'}")
    print(f"expected score: ~{expected_score:.3f} (from V3 frozen run)")

    assert abs(mean - expected_score) < 0.05, (
        f"\n  judge stability broken on {item_id!r}:"
        f"\n    expected mean ~{expected_score:.3f} (from V3 frozen run)"
        f"\n    observed mean  {mean:.3f}"
        f"\n    scores:        {scores}"
        f"\n  the judge has drifted — re-investigate against new data."
    )
