# Background

If you've written automated tests but never tested an LLM, this page is
for you. The job is the same as any other test suite: send an input,
check the output. The catch is that the output is a sentence, not a
value, so `output == expected` almost never works. Everything below is
about how to write a useful assertion when the answer is prose.

## The vocabulary, translated

| In this repo | What a QA engineer would call it |
|---|---|
| `ExactMatchScorer` | `assert output == expected`. Fine for IDs and codes, breaks on sentences. |
| `BleuScorer` / `RougeScorer` | A fuzzy string diff. Counts how many words the model's answer shares with the expected answer. |
| `SemanticScorer` | An assertion with a tolerance, where the tolerance is "close enough in meaning." Implemented by turning each sentence into a vector of numbers and checking how close the vectors are. |
| `LLMJudgeScorer` | A second AI reads the answer and grades it against a written rubric. Convenient, but the grader can be wrong or biased. |
| `data/golden_set.yaml` | The test fixtures. Each item has a question and one example of a correct answer. The example is a reference, not the only acceptable wording. |
| `data/human_labels.yaml` | A frozen snapshot of model outputs plus the human PASS or FAIL grade for each one. Lets a failing test mean "the scorer is broken," not "the model said something different today." |
| `tests/test_calibration.py` | Tests that test the test tooling. Does the scorer agree with the human grader? |
| `xfail(strict=True, reason=...)` | A "known bug" marker. The test is allowed to fail, but if it ever starts passing, the build breaks so you know the bug is gone. |
| `tests/test_bias.py` | A metamorphic test. Change one detail in the question (a name, a date) and check the answer changes the same way. |
| `tests/test_judge_variance.py` | A flakiness test. Run the same grading 5 times, check the score doesn't bounce around. |
| `mocked` vs `ollama` markers | Unit tests with a fake HTTP layer (fast, no model needed) vs integration tests against a real local model. |
| `eval_harness/providers/` | The only code that talks to the model over HTTP. Tests mock at this boundary. |

LLM testing is regular software testing with two extra problems: the
output is prose, and the thing scoring the prose can itself be wrong.

## What each scorer gets wrong

Read the legend first: ✅ right, ❌ wrong, ⚠️ fragile (sometimes
right, sometimes not), `n/a` (not applicable). `(F#)` points to a
numbered finding listed below the table.

| Kind of answer | Exact | BLEU | ROUGE | Semantic | Judge |
|---|---|---|---|---|---|
| Right answer wrapped in prose (`"Paris"` vs `"The capital of France is Paris."`) | ❌ (F1) | ❌ (F10) | ❌ (F10) | ⚠️ (F2) | ✅ |
| Right answer in a different shape (8 planets listed vs `"8"`) | ❌ | ❌ | ❌ | ❌ (F3) | ✅ (F4) |
| Prose vs prose, overlapping words | ❌ | ✅ | ✅ | ✅ | ✅ |
| Same idea, different words | ❌ | ❌ | ❌ | ✅ | ✅ |
| Confident hallucination | ✅ wrong reason | ✅ by accident | ✅ by accident | ⚠️ may pass | ❌ self-grading bias (F4, F8) |
| Name swap (David vs Priya) | n/a | n/a | n/a | ✅ (F7) | n/a |
| Refusal ("I don't know") | ❌ | ❌ | ❌ | ❌ | ⚠️ untested |

No column is all green. That's why the harness runs more than one
scorer and checks they agree. BLEU and ROUGE are not stricter versions
of exact match; whether they help depends on the shape of the
reference text, not on the metric itself (F10).

## Findings

Each finding is a thing the calibration run actually showed.

- **F1.** Strict `==` fails on almost every prose answer.
- **F2.** A clean paraphrase scored 0.725 on semantic similarity, just under a "reasonable looking" 0.75 threshold. The threshold felt safe and was wrong.
- **F3.** Semantic similarity measures "do these sentences sound alike," not "is this answer correct." Two wrong-but-similar answers score high.
- **F4.** When the model grades its own work, it gives itself the benefit of the doubt right at the pass/fail line.
- **F5.** The judge's written reasoning and its numeric score don't always match. The reasoning says "wrong" and the score says 0.8.
- **F6.** Marking known failures with `xfail(strict=True)` turns calibration into a contract. If the bug ever silently fixes itself, the build breaks and you find out.
- **F7.** Swapping a name in a question (David to Priya) changed the answer in 1 of 4 pairs. Bias-style tests catch this; semantic similarity flagged the drift.
- **F8.** At the threshold edge, the judge is not noisy (random around 0.75). It is stuck (always 0.75). Re-running won't save you.
- **F9.** No length bias was detected on `llama3.2`. A useful "nothing is broken here" result, not a failure.
- **F10.** Whether BLEU and ROUGE help depends on how the reference answer is written, not on the metric. A short reference makes them fail; a long one makes them work.

## Why a few choices look odd

**All scorers are async, even the ones that don't do any I/O.** This
harness only runs inside pytest, where everything is async already.
A real library (DeepEval, Ragas) would offer both sync and async
versions to avoid forcing one onto the caller.

**Calibration uses frozen model outputs, not live ones.** A red test
should mean "the scorer is broken," not "the model felt different
today." For drift detection, run the tests marked `ollama` against a
live model.

**No Cohen's kappa.** Kappa is the standard metric for inter-rater
agreement. With 10 items and one human grader, the confidence interval
is roughly negative 0.3 to positive 0.9, which is too wide to mean
anything. Per-item `xfail(strict=True)` carries more information and
breaks the build if a finding silently disappears.

**The dataset has 18 items on purpose.** The first 10 already produced
the findings. Adding more items would tighten the percentages without
teaching anything new. A real release-gating eval needs hundreds.
