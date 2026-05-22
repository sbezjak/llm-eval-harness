# Walkthrough

Live HTML test report: [report.html](https://sbezjak.github.io/llm-eval-harness/reports/report.html)
· Coverage: [coverage](https://sbezjak.github.io/llm-eval-harness/reports/coverage/)

## Why this project exists

I'm an automation tester. My usual job is checking that an app does the
same thing every time. AI testing is the opposite: the same question can
produce a different answer on every run, and more than one of those
answers can be correct. The job becomes "write a useful assertion when
the answer is prose."

This repo is my attempt to do that the way a QA engineer would: build a
small fixture (10 questions with hand-written expected answers), run a
local model (`llama3.2` via Ollama), grade the outputs by hand, then
write five different automated scorers and measure how often each one
agrees with the human grade. The findings are the disagreements.

### Vocabulary, translated

If you've never written an LLM test before, here is the rough mapping
from this repo to language a QA engineer already knows.

| In this repo | What a QA engineer would call it |
|---|---|
| `ExactMatchScorer` | `assert output == expected`. Fine for IDs and codes, breaks on sentences. |
| `BleuScorer` / `RougeScorer` | A fuzzy string diff. Counts how many words the model shares with the expected answer. |
| `SemanticScorer` | An assertion with a tolerance, where the tolerance is "close enough in meaning." Done by turning each sentence into a vector and checking the angle between vectors. |
| `LLMJudgeScorer` | A second AI reads the answer and grades it against a written rubric. Convenient, but the grader can be wrong or biased. |
| `data/golden_set.yaml` | The test fixtures. Each item has a question and one example of a correct answer. |
| `data/human_labels.yaml` | A frozen snapshot of model outputs plus the human PASS/FAIL grade. A failing test means "the scorer is broken," not "the model said something different today." |
| `xfail(strict=True, reason=...)` | A "known bug" marker. The test is allowed to fail, but if it ever starts passing, the build breaks on purpose. |
| `mocked` vs `ollama` markers | Unit tests with a fake HTTP layer (fast, no model needed) vs integration tests against a real local model. |

LLM testing is regular software testing with two extra problems: the
output is prose, and the thing scoring the prose can itself be wrong.

## TL;DR (5 minutes)

**What this is.** A pytest harness that runs five scorers
(`ExactMatch`, `BLEU`, `ROUGE`, `Semantic`, `LLMJudge`) against a
10-item golden set on a local `llama3.2` model, calibrated against
human PASS/FAIL grades. Every known scorer limitation is locked in as
`xfail(strict=True)` with a written reason.

**What this isn't.** A production eval framework. The dataset is small
(10 items) and a single model is tested. Findings are reproducible on
this model, not universal claims about LLM judges.

**Three headline findings, in priority order.**

1. **The judge passes its own hallucination, deterministically** (F4 +
   F8). The model invented a fake pytest flag. The same model acting
   as judge scored its own wrong answer at 0.700, exactly on the pass
   threshold, every one of 5 reruns. A flaky test gets investigated; a
   deterministic wrong-pass ships the bug. Averaging more runs does
   nothing when there is no noise to average out.
2. **BLEU and ROUGE are not "bad at prose," they need matching
   reference shape** (F10). When the expected answer is "8" and the
   model returns a paragraph, n-gram overlap collapses to zero, not
   because the answer is wrong but because the texts are different
   lengths. Same metric, longer reference, score works again.
3. **`xfail(strict=True)` with a reason string is the load-bearing
   pattern** (F6). Every "scorer disagrees with the human" case is
   pinned as an expected failure with the structural reason in the
   message. If a scorer's known weakness ever disappears, the suite
   breaks loudly and forces re-investigation. The xfail set is the
   spec for what each scorer is known to get wrong.

**The pattern to lift.** `pytest.xfail(strict=True, reason="...")` is
not LLM-specific. It works on any flaky integration where the failure
mode is understood. Use it whenever a test fails for a known reason
that you want documented in code and tripwired to break on silent
recovery.

### Failure-mode matrix (at a glance)

| Kind of answer | Exact | BLEU | ROUGE | Semantic | Judge |
|---|---|---|---|---|---|
| Right answer wrapped in prose (`"Paris"` vs `"The capital is Paris."`) | FAIL (F1) | FAIL (F10) | FAIL (F10) | fragile (F2) | PASS |
| Right answer in a different shape (8 planets listed vs `"8"`) | FAIL | FAIL | FAIL | FAIL (F3) | PASS (F4) |
| Same idea, different words | FAIL | FAIL | FAIL | PASS | PASS |
| Confident hallucination | PASS for wrong reason | PASS by accident | PASS by accident | may pass | FAIL of self-grading (F4, F8) |
| Name swap (David vs Priya) | n/a | n/a | n/a | PASS (F7) | n/a |

No column is all green. The point of the harness is that the scorers
disagree, and the disagreements are the data.

## Findings

Each finding follows the same shape: plain-English question, the test
that asks it, a short log snippet that shows the answer, one paragraph
on what an automation tester would take away.

### F1. Strict `==` fails on almost every prose answer

**Question.** If the model answers correctly but wraps the answer in a
full sentence, does `output == expected` still work?

**Test.** `tests/test_calibration.py::test_exact_match_calibration` (1
PASS, 9 XFAIL across the 10-item golden set).

```
item:           factual_002
question:       What is the chemical symbol for silver?
expected:       'Ag'
frozen_output:  'The chemical symbol for silver is Ag.'
human_verdict:  PASS
scorer:         exact_match
scorer passed:  False
scorer reason:  outputs differ
```

It's not just this one item. The same pattern appears almost
everywhere in the golden set:

```
expected: '8'         output: 'There are 8 planets...'       FAIL
expected: 'Ag'        output: 'The chemical symbol is Ag.'   FAIL
expected: 'Paris'     output: 'The capital is Paris.'        FAIL
```

9 of the 10 calibration items fail exact match for exactly this
reason. The one that passes (`procedural_001`) only passes by accident:
the model hallucinated a wrong answer, the human also said FAIL, so
"both said FAIL" counts as agreement.

**Takeaway.** Exact match is fine for IDs, status codes, and enum
values. It breaks the moment the system under test returns a sentence.
This is not news, but it sets the floor: if you only have exact match,
you cannot test an LLM. Every other scorer in this repo exists to fix
this one problem in a different way.

### F2. A "reasonable looking" cosine threshold rejected a clean paraphrase

**Question.** If I set the semantic-similarity threshold to 0.75
because that's what tutorials use, am I safe?

**Test.** `tests/test_semantic_scorer.py::test_paraphrase_scores_higher_than_unrelated`.
The paraphrase ("The capital of France is Paris" vs "Paris") scored
~0.725, which is below 0.75.

```
paraphrase score: 0.725
threshold:        0.75
verdict:          FAIL
```

**Takeaway.** Picking a threshold by feel ("0.75 sounds right") is a
trap. You need calibration data: known-correct paraphrases on one
side, known-wrong answers on the other, and a threshold chosen from
where the distributions actually sit. The number that "feels safe"
will reject right answers.

### F3. Semantic similarity measures "sound alike," not "is correct"

**Question.** If I lower the cosine threshold until the right answers
pass, are the wrong answers still caught?

**Test.** `tests/test_calibration.py::test_semantic_calibration` (5
PASS, 5 XFAIL). On the planets question, the lowest right-answer score
and the only wrong-answer score in the set sat 0.004 apart.

```
right answer (8 planets listed): cosine 0.194
wrong answer about planets:      cosine 0.198
```

**Takeaway.** Semantic similarity is textual proximity, not factual
correctness. Two answers about the same topic land in the same
neighborhood whether one is right and the other is wrong. There is no
threshold that admits all the right ones without also admitting the
wrong one. Useful as one signal in a panel; not enough on its own.

### F4. The judge passes its own hallucination (self-grading bias)

**Question.** If the same model writes the answer and grades the
answer, does the grading catch the model's own mistakes?

**Test.** `tests/test_calibration.py::test_judge_calibration[procedural_001]`
(9 PASS, 1 XFAIL). The model invented `--junit-xml-filter`, a pytest
flag that doesn't exist. The judge (same `llama3.2`) gave it
correctness 8/10, relevance 6/10, combined 0.700.

```
item:           procedural_001
question:       <pytest filtering question>
model output:   <answer using fake --junit-xml-filter flag>
judge score:    0.700  (passes the 0.700 threshold)
human verdict:  FAIL
```

**Takeaway.** The judge is the same model that wrote the bad answer,
so the hallucinated flag doesn't look wrong to either of them. This is
the canonical LLM-as-judge failure mode and the headline finding of
the project. Fix is a different (stronger) judge model, or a panel of
judges, or a non-LLM check for factual claims like flag names that can
be looked up.

### F5. The judge's written reasoning can contradict its numeric score

**Question.** The judge returns two things: a paragraph explaining its
grading, and a number. If I'm building a UI or a debug log around this,
which one do I trust?

**Background, since this finding only makes sense once you know how the
judge works.** The judge is a second model call. We send it the
question, the original model's answer, and a rubric ("score correctness
out of 10, score relevance out of 10, explain your reasoning, then give
the scores"). It returns JSON with three fields: a `reasoning` string, a
`correctness` integer, and a `relevance` integer. The numeric score we
gate on is `(correctness + relevance) / 20`. The `reasoning` string is
not used by the assertion at all; it's only there so a human reading
the report can see why the judge scored the way it did.

**Test.** Per-item evidence captured in `tests/test_eval_pipeline.py`
and `tests/test_calibration.py`. Run with `-s` to stream the judge's
full reasoning string into stdout for every item, then read the
"Captured stdout" panel in the HTML report.

**What the inconsistency looks like.** Several items showed reasoning
that pointed clearly at "this answer is wrong" while the numeric score
still landed above the 0.7 pass threshold. A simplified example of the
shape (paraphrased from the reports):

```
judge reasoning: "The model's answer mentions --junit-xml-filter, which
                 is not a real pytest flag. The example would not run.
                 Correctness is low."
judge correctness: 8
judge relevance:   6
combined score:    0.700  -> PASS
```

The prose says "low correctness." The number says 8/10. They are not
consistent, and the assertion uses the number.

**Takeaway for an automation tester.** Treat the judge's written
explanation as a debugging hint, never as the assertion. Two practical
consequences:

1. **Do not put the judge's reasoning string into a user-facing UI as
   evidence for the verdict.** Users (and reviewers) will trust the
   prose more than the number, and the prose is the part that can lie
   to you. If you must surface something, surface the rubric breakdown
   (`correctness=8, relevance=6, threshold=0.7`), not the paragraph.
2. **If you find this kind of contradiction, it's a signal that the
   judge isn't following the rubric carefully on that item.** It does
   not mean the judge is broken in general, but it does mean that
   particular score is unreliable. Flag the item for human review;
   don't try to "fix" the judge by tweaking the rubric until the prose
   matches the number, because that's optimizing the wrong direction.

### F6. `xfail(strict=True)` turned the suite into a tripwire

**Question.** How do you write down "this is a known limitation" in a
way that breaks the build if the limitation silently disappears?

**Test.** Every xfail in `tests/test_calibration.py`,
`tests/test_eval_pipeline.py`, `tests/test_bias.py`, and
`tests/test_judge_variance.py`. Each one carries `strict=True` plus a
`reason=` describing the exact structural property that caused the
disagreement.

```
test_calibration.py::test_exact_match_calibration[factual_001]  XFAIL
  reason: exact match wrongly says FAIL: model returns the right answer
  wrapped in conversational prose, so output != expected (Finding 1)
```

**Takeaway.** This pattern works in any test suite, AI or not.
`xfail(strict=True, reason="<finding>")` does three jobs at once: it
keeps the suite green while documenting reality, it serves as in-code
documentation right next to the assertion, and it tripwires silent
recovery. The day a scorer "fixes itself," the XPASS breaks the build
and somebody has to update the docs instead of quietly losing the
finding. Two findings in this project (F8 stuck-at-threshold, F10
reference-shape) only surfaced because the strict xfails were wrong
about the failure mode and the suite refused to stay green about it.

### F7. A name swap caused output drift in 1 of 4 bias pairs

**Question.** If I change only a name (David to Priya) in a question,
does the model give the same advice?

**Test.** `tests/test_bias.py::test_bias_pair_outputs_are_similar[career_advice_gender]`
XFAIL. Cosine 0.667 against a 0.70 threshold. Three other pairs passed.

```
pair:        career_advice_gender
output_a:    <David's answer: 7 numbered points, motivational closer>
output_b:    <Priya's answer: 10 numbered points, "set realistic goals">
similarity:  0.667  (threshold 0.70)
verdict:     FAIL - outputs drift
```

**Takeaway.** n=4 is a smoke alarm, not a verdict. The structural
drift is reproducible, so it's worth logging, but the cause is
uncertain (could be gender-coded, name-association, or just
output-length variance). The lesson is the testing pattern, not the
bias claim: identical inputs except for one variable should produce
identical-shape outputs, and divergence is the signal. Real bias work
uses BBQ, BOLD, or similar benchmarks with hundreds of pairs.

### F8. At the threshold edge, the judge is stuck, not noisy

**Question.** I expected the judge to be noisy at temp=0. Is it?

**Test.** `tests/test_judge_variance.py::test_judge_stability[procedural_001-0.7]`.
Ran the judge 5 times on the same frozen input.

```
run 1: score=0.700 passed=True
run 2: score=0.700 passed=True
run 3: score=0.700 passed=True
run 4: score=0.700 passed=True
run 5: score=0.700 passed=True

mean: 0.700  stdev: 0.000  verdict flip: no
```

**Takeaway.** This is worse than noisy. A noisy judge eventually flips
the verdict and someone investigates the red CI; a stuck judge stays
green every run and the bug ships forever. Critically, averaging more
runs does nothing when there is no noise to average out. If your
defense against judge errors is "we'll just rerun," this finding kills
that defense for stuck cases. The fix has to be structural (different
judge model, panel of judges, factual check for the specific claim).

### F9. No length bias detected on `llama3.2` (null result)

**Question.** LLM judges often score longer answers higher. Does this
one?

**Test.** `tests/test_length_bias.py`. Three questions, each with a
short and a long correct answer. The delta long_score minus short_score
is printed per pair.

```
short (  6 chars): score=1.000 passed=True
long  (432 chars): score=1.000 passed=True
delta (long - short): +0.000
  -> no meaningful difference on this case
```

**Takeaway.** A useful null result, with a sized caveat. n=3 cannot
support "this judge has no length bias in general." It can support "on
these 3 short-factual cases, the judge scored both versions
identically." Mild evidence that the rubric design (separate
correctness/relevance, reasoning before score) suppresses length bias
on simple questions. The pattern is more important than the result:
when you suspect a bias, pair-test for it, even at small n, because
"checked and found nothing" is the floor for "no detectable bias."

### F10. BLEU/ROUGE depend on reference shape, not on the metric

**Question.** "BLEU and ROUGE are bad at prose" is the usual one-liner.
Is that the real story?

**Test.** `tests/test_calibration.py::test_bleu_calibration` (3 PASS,
7 XFAIL) and `test_rouge_calibration` (4 PASS, 6 XFAIL). The PASS rows
are the items where the expected reference happens to be a full
sentence; the XFAIL rows are items where the reference is a single
token like "8" or "Ag." Same metric, different references, opposite
result.

```
expected: 'Ag'
output:   'The chemical symbol for silver is Ag.'
BLEU score: ~0.0   (no 2-gram or 3-gram overlap possible vs a 1-token reference)
ROUGE score: ~0.0  (denominator is the long answer length)
```

**Takeaway.** BLEU and ROUGE work when the reference and the output
are similar in shape and length. Short reference plus long answer
collapses the score; long reference plus long answer recovers it.
Practical version: if you want BLEU or ROUGE to mean anything, write
reference answers in roughly the shape you expect the system to
return. For one-word answers, use exact match or a judge instead.

## Limitations

**The dataset is 10 items on purpose.** The first 10 produced every
finding above. Adding more items would tighten percentages without
teaching anything new. A real release-gating eval needs hundreds of
items across multiple raters; dataset construction alone is a
multi-day task at that scale.

**Only one model is tested (`llama3.2`).** Every finding is reproducible
on this model. Whether "judge passes its own hallucination" holds on
GPT-4o or Claude is a question this repo does not answer. The pattern
(test the judge separately from the system under test) transfers; the
specific numbers do not.

**Bias check is n=4, length-bias check is n=3.** Both are smoke alarms,
not verdicts. Real bias work uses BBQ, BOLD, or paired-prompt
benchmarks with hundreds of items across multiple demographic axes.
Calling either of these "evidence the model is unbiased" would be
overclaiming.

**No Cohen's kappa for inter-rater agreement.** Cohen's kappa is a
single number between -1 and 1 that answers the question "do two
graders agree more than they would by chance?" 1 means perfect
agreement, 0 means they agree only as often as random guessing would
predict, negative numbers mean they disagree more than chance. In LLM
evaluation, the standard use is: one grader is the human, the other
grader is the automated scorer, and a high kappa means the scorer is a
faithful stand-in for human judgment. With only 10 items and one human
grader, the statistical confidence interval around any kappa value
this small dataset would produce is enormous (roughly -0.3 to +0.9),
which means the number itself would be uninformative. Per-item
`xfail(strict=True)` with a structural reason carries more information
than a kappa from too-thin data: it names *which* items disagree and
*why*. If this project grew to hundreds of items and multiple human
graders, kappa would become the right metric to track.

**All scorers are async, even the ones that do no I/O.** This harness
runs only inside pytest, where the event loop already exists, so a
single async interface keeps callsites uniform. Real libraries
(DeepEval, Ragas) offer both sync and async interfaces so callers
aren't forced into one. The trade-off taken here: callsite simplicity
in exchange for being unusable outside an async context.

**Calibration uses frozen model outputs, not live ones.** Deliberate:
a red calibration test must mean "the scorer is broken," not "the
model felt different today." For drift detection against the live
model, the `ollama`-marked tests in `test_eval_pipeline.py` do that
separately.

**What a production-grade version would look like.** A stronger judge
model (or a panel of judges grading the same item, then aggregated) so
self-grading bias is visible against a baseline that doesn't share the
same weights as the system under test. Multiple human raters from the
start so the calibration set can support a real Cohen's kappa.
Dataset of at least 50 items per category (factual, definition,
procedural, reasoning) so percentages actually mean something and so
distributions of right-vs-wrong scores can be plotted rather than
counted. The patterns in this harness scale up; the numbers in it do
not.