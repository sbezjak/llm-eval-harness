# Human eval — round 1

Your read of each item from the V1 pytest report. Pre-filled from
`reports/V1_pytest.html`. Only **my verdict** and **note** need you.

How to fill in:

- **my verdict**: would you, as a human grader, accept this answer? PASS or FAIL.
- **note**: anything weird — wrong-but-close, right-but-wordy, hallucinated, etc.

The interesting columns are the last two. If your verdict is PASS but
semantic says FAIL → threshold too strict on that item. If semantic says
PASS but you'd say FAIL → threshold too loose.

---

## factual_001 — capital of Slovenia

- **question:** What is the capital of Slovenia?
- **expected:** `Ljubljana`
- **got:** `The capital of Slovenia is Ljubljana.`
- **exact:** FAIL
- **semantic:** PASS
- **score:** 0.827
- **my verdict:** PASS
- **note:** Got the answer right even if it did include more words.

## factual_002 — chemical symbol for silver

- **question:** What is the chemical symbol for silver?
- **expected:** `Ag`
- **got:** `The chemical symbol for silver is Ag.`
- **exact:** FAIL
- **semantic:** FAIL
- **score:** 0.583
- **my verdict:** PASS
- **note:** Got the answer right even if it did include more words.

## factual_003 — continent of Lima

- **question:** In what continent is Lima located?
- **expected:** `South America`
- **got:** `Lima is the capital city of Peru, and it is located in South America.`
- **exact:** FAIL
- **semantic:** FAIL
- **score:** 0.569
- **my verdict:** PASS
- **note:** Got the answer right even if it did include more words.

## factual_004 — number of planets

- **question:** How many planets are in our solar system?
- **expected:** `8`
- **got:** `There are eight planets in our solar system:\n1. Mercury\n2. Venus\n3. Earth\n4. Mars\n5. Jupiter\n6. Saturn\n7. Uranus\n8. Neptune\n(Note: Pluto was reclassified in 2006...)`
- **exact:** FAIL
- **semantic:** FAIL
- **score:** 0.194
- **my verdict:** PASS
- **note:** Got the answer right even if it did include more words.

## definition_001 — AI acronym

- **question:** What does the acronym AI stand for?
- **expected:** `Artificial Intelligence`
- **got:** `The acronym "AI" stands for Artificial Intelligence.`
- **exact:** FAIL
- **semantic:** FAIL
- **score:** 0.551
- **my verdict:** PASS
- **note:** Got the answer right even if it did include more words.

## definition_002 — black hole

- **question:** In one sentence, what is a black hole?
- **expected:** `A region of space where the gravitational pull is so strong that nothing, not even light, can escape from it.`
- **got:** `A black hole is a region in space where the gravitational pull is so strong that nothing, including light, can escape from it, making it invisible and detectable only through its effects on surrounding matter.`
- **exact:** FAIL
- **semantic:** FAIL
- **score:** 0.597
- **my verdict:** PASS
- **note:** Right answer, different wording.

## procedural_001 — pytest -m

- **question:** Which pytest command runs tests marked either 'slow' or 'network'?
- **expected:** `pytest -m 'slow or network'`
- **got:** ```pytest --durations=0 --junit-xml=/path/to/output.xml --junit-xml-filter='slow|network'``` (note: this is **not a real pytest flag** — model hallucinated)
- **exact:** FAIL
- **semantic:** FAIL
- **score:** 0.579
- **my verdict:** FAIL
- **note:** Completely wrong.

## reasoning_001 — cold vs warm blooded

- **question:** What is the difference between cold blooded and warm blooded animals?
- **expected:** `Cold blooded animals rely on external sources for body heat, while warm blooded animals maintain a constant internal body temperature.`
- **got:** `The terms "cold-blooded" and "warm-blooded" are outdated... Ectothermic animals rely on external sources of heat to regulate their body temperature... Endothermic animals generate heat internally and maintain a relatively constant body temperature...` (long, ~400 words, with examples)
- **exact:** FAIL
- **semantic:** PASS
- **score:** 0.767
- **my verdict:** PASS
- **note:** Got the answer right even if it did include more words.

## reasoning_002 — why ice floats

- **question:** In one sentence, why does ice float on water?
- **expected:** `Ice floats on water because it is less dense than liquid water.`
- **got:** `Ice floats on water because it is less dense than liquid water, meaning that the molecules in ice are arranged in a crystalline structure that takes up more space than the molecules in liquid water.`
- **exact:** FAIL
- **semantic:** PASS
- **score:** 0.908
- **my verdict:** PASS
- **note:** Same first sentence.

## reasoning_003 — photosynthesis

- **question:** How does photosynthesis work in plants?
- **expected:** `Photosynthesis in plants converts light energy into chemical energy, using sunlight to transform carbon dioxide and water into glucose and oxygen.`
- **got:** `Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of organic compounds, such as glucose...` (long, multi-step explanation with sections)
- **exact:** FAIL
- **semantic:** PASS
- **score:** 0.860
- **my verdict:** PASS
- **note:** Different wording.

---

## Summary (fill in after the table)

- **Items I'd pass as a human:** 9 / 10
- **Items semantic passed:** 4 / 10  (factual_001, reasoning_001, reasoning_002, reasoning_003)
- **Items exact passed:** 0 / 10
- **Disagreements (semantic FAIL but I'd PASS):** 8
- **Disagreements (semantic PASS but I'd FAIL):** 0

If the first count is high → threshold is too strict, lower it.
If the second is high → threshold is too loose, raise it.
If both are low → threshold is fine; the wins from semantic are real.

---

## Score distribution at a glance

```
factual_001  0.827  PASS  ✓
reasoning_002 0.908  PASS  ✓
reasoning_003 0.860  PASS  ✓
reasoning_001 0.767  PASS  ✓ (just barely)
─────────────  0.750  threshold ─────────────
definition_002 0.597  FAIL
factual_002   0.583  FAIL
procedural_001 0.579  FAIL  (model actually wrong)
factual_003   0.569  FAIL
definition_001 0.551  FAIL
factual_004   0.194  FAIL  (huge list vs "8")
```

The gap between the lowest pass (0.767) and the highest fail (0.597) is
0.17 — narrow. Worth noting that procedural_001, the only item where the
model is actually wrong, sits at 0.579 — right in the middle of the
"right but failed" cluster. No threshold separates right from wrong here.
That's the lesson worth chewing on once you've graded.
