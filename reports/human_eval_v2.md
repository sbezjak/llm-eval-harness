# Human eval — round 2 (with judge)

V1 columns are pre-filled (model is `llama3.2` at `temperature=0`, so the
`got` text is effectively identical to V1). The judge columns are
pre-filled from `reports/V2_pytest.html`. **All you need to do is fill
in the `judge note` field for each row** — your read of whether the
judge's score and reasoning make sense.

What to look for in the judge note:

- Did the judge's *reasoning text* match its *score*? On factual_003,
  reasoning_001, and procedural_001 the judge wrote "did not directly
  address the question" even though the answer did — Finding 5
  territory.
- For procedural_001 specifically: the judge gave 8/10 correctness for
  a fully hallucinated answer. Self-grading bias — Finding 4.

The interesting columns are `judge passed` (vs `my verdict`) and
`judge note`.

---

## factual_001 — capital of Slovenia

- **question:** What is the capital of Slovenia?
- **expected:** `Ljubljana`
- **got:** `The capital of Slovenia is Ljubljana.`
- **exact:** FAIL
- **semantic:** PASS (0.827)
- **my verdict:** PASS
- **judge correctness:** 10 /10
- **judge relevance:** 10 /10
- **judge score:** 1.000
- **judge passed:** PASS
- **judge note:**

## factual_002 — chemical symbol for silver

- **question:** What is the chemical symbol for silver?
- **expected:** `Ag`
- **got:** `The chemical symbol for silver is Ag.`
- **exact:** FAIL
- **semantic:** FAIL (0.583)
- **my verdict:** PASS
- **judge correctness:** 10 /10
- **judge relevance:** 10 /10
- **judge score:** 1.000
- **judge passed:** PASS
- **judge note:**

## factual_003 — continent of Lima

- **question:** In what continent is Lima located?
- **expected:** `South America`
- **got:** `Lima is the capital city of Peru, and it is located in South America.`
- **exact:** FAIL
- **semantic:** FAIL (0.569)
- **my verdict:** PASS
- **judge correctness:** 8 /10
- **judge relevance:** 10 /10
- **judge score:** 0.900
- **judge passed:** PASS
- **judge reasoning:** *"correctly identifies Lima as being in South America, but does not directly address the question of which continent it is located on."*
- **judge note:** Reasoning is wrong — the answer DID directly say "South America." Score landed roughly right (0.9, passes) but the prose is post-hoc filler.

## factual_004 — number of planets

- **question:** How many planets are in our solar system?
- **expected:** `8`
- **got:** `There are eight planets in our solar system:\n1. Mercury\n2. Venus\n...\n(Note: Pluto was reclassified in 2006...)`
- **exact:** FAIL
- **semantic:** FAIL (0.194)
- **my verdict:** PASS
- **judge correctness:** 10 /10
- **judge relevance:** 10 /10
- **judge score:** 1.000
- **judge passed:** PASS
- **judge note:** **The judge rescued this from semantic 0.194 → 1.000.** This is the single empirical case for putting the judge in the suite at all — the hybrid rubric reads past textual shape.

## definition_001 — AI acronym

- **question:** What does the acronym AI stand for?
- **expected:** `Artificial Intelligence`
- **got:** `The acronym "AI" stands for Artificial Intelligence.`
- **exact:** FAIL
- **semantic:** FAIL (0.551)
- **my verdict:** PASS
- **judge correctness:** 10 /10
- **judge relevance:** 10 /10
- **judge score:** 1.000
- **judge passed:** PASS
- **judge note:**

## definition_002 — black hole

- **question:** In one sentence, what is a black hole?
- **expected:** `A region of space where the gravitational pull is so strong that nothing, not even light, can escape from it.`
- **got:** `A black hole is a region in space where the gravitational pull is so strong that nothing, including light, can escape from it, making it invisible and detectable only through its effects on surrounding matter.`
- **exact:** FAIL
- **semantic:** FAIL (0.597)
- **my verdict:** PASS
- **judge correctness:** 9 /10
- **judge relevance:** 10 /10
- **judge score:** 0.950
- **judge passed:** PASS
- **judge note:**

## procedural_001 — pytest -m  ⚠ (the centerpiece)

- **question:** Which pytest command runs tests marked either 'slow' or 'network'?
- **expected:** `pytest -m 'slow or network'`
- **got:** `pytest --durations=0 --junit-xml=/path/to/output.xml --junit-xml-filter='slow|network'` (note: hallucinated flags)
- **exact:** FAIL
- **semantic:** FAIL (0.579)
- **my verdict:** FAIL
- **judge correctness:** 8 /10
- **judge relevance:** 6 /10
- **judge score:** 0.700
- **judge passed:** PASS  ⚠ (at threshold edge)
- **judge reasoning:** *"The model provided multiple correct ways to run the test, but failed to directly answer the question. The model's answer is partially relevant as it does provide a solution, but could be more concise..."*
- **judge note:** **Self-grading bias caught red-handed.** The "multiple correct ways" the judge sees are all hallucinated flags. Same llama3.2 grading its own hallucination as plausible. This is the empirical hook for project 5 (cross-model judging). Note also: even though the judge passed it, this WAS the lowest score in the distribution — the judge has *some* signal, just not enough to push the wrong answer below the threshold.

## reasoning_001 — cold vs warm blooded

- **question:** What is the difference between cold blooded and warm blooded animals?
- **expected:** `Cold blooded animals rely on external sources for body heat, while warm blooded animals maintain a constant internal body temperature.`
- **got:** Long ~400-word explanation of ectothermic vs endothermic regulation.
- **exact:** FAIL
- **semantic:** PASS (0.767)
- **my verdict:** PASS
- **judge correctness:** 8 /10
- **judge relevance:** 9 /10
- **judge score:** 0.850
- **judge passed:** PASS
- **judge reasoning:** *"clear explanation… However, it did not directly address the question in its initial response."*
- **judge note:** Same "did not directly address the question" tic as factual_003 and procedural_001. Answer literally addressed the question. Three items, same wrong rationalization → systematic, not random.

## reasoning_002 — why ice floats

- **question:** In one sentence, why does ice float on water?
- **expected:** `Ice floats on water because it is less dense than liquid water.`
- **got:** `Ice floats on water because it is less dense than liquid water, meaning that the molecules in ice are arranged in a crystalline structure that takes up more space than the molecules in liquid water.`
- **exact:** FAIL
- **semantic:** PASS (0.908)
- **my verdict:** PASS
- **judge correctness:** 10 /10
- **judge relevance:** 10 /10
- **judge score:** 1.000
- **judge passed:** PASS
- **judge note:**

## reasoning_003 — photosynthesis

- **question:** How does photosynthesis work in plants?
- **expected:** `Photosynthesis in plants converts light energy into chemical energy, using sunlight to transform carbon dioxide and water into glucose and oxygen.`
- **got:** Long multi-step explanation with sections.
- **exact:** FAIL
- **semantic:** PASS (0.860)
- **my verdict:** PASS
- **judge correctness:** 8 /10
- **judge relevance:** 9 /10
- **judge score:** 0.850
- **judge passed:** PASS
- **judge note:**

---

## Summary

- **Items I'd pass as a human:** 9 / 10
- **Items semantic passed:** 4 / 10  (factual_001, reasoning_001, reasoning_002, reasoning_003)
- **Items judge passed:** 10 / 10  ← every item, including the wrong one
- **Disagreements (judge PASS, semantic FAIL):** 6  ← the rescues; the reason the judge is in the suite
- **Disagreements (judge FAIL, semantic PASS):** 0
- **Items judge passed that I'd FAIL:** 1  (procedural_001)  ← self-grading bias, exactly the predicted failure mode

The hybrid rubric is doing its job (6 rescues, including factual_004
which was a textbook shape-mismatch). The cost is one false positive
from self-grading bias on the only adversarial item — and that false
positive sat *exactly* at the threshold (0.700 vs 0.700), suggesting
the judge had partial signal but not enough resolution to push it
below.

---

## Score distribution

```
factual_001     judge 1.000  semantic 0.827
factual_002     judge 1.000  semantic 0.583
factual_004     judge 1.000  semantic 0.194  ← shape mismatch rescue
definition_001  judge 1.000  semantic 0.551
reasoning_002   judge 1.000  semantic 0.908
definition_002  judge 0.950  semantic 0.597
factual_003     judge 0.900  semantic 0.569
reasoning_001   judge 0.850  semantic 0.767
reasoning_003   judge 0.850  semantic 0.860
─────────────  judge threshold 0.70 ─────────────
procedural_001  judge 0.700  semantic 0.579   ← model is wrong
```

What this shows:

- The judge column **does** separate procedural_001 (the only wrong
  item) — it's at the bottom. Semantic couldn't do this; every
  threshold that passed the lowest right answer also passed the
  wrong one.
- The judge clusters on round numbers: scores are 0.70 / 0.85 / 0.90 /
  0.95 / 1.00. Effective resolution is ~5 buckets. Predicted
  small-model rubric ceiling — note in article.md.
- Judge resolution is *just barely* not enough to fail procedural_001.
  Threshold 0.71 would catch it but would also start failing the
  reasoning items at 0.85, which would be wrong. The fix is a better
  judge model, not a better threshold (Finding 4).
