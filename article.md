# Article notes

Running notebook of findings worth keeping for the writeup. Each entry is
self-contained: what was tried, what happened, why it matters. Add to it
as the project progresses.

---

## Finding 1 — Strict exact match fails on almost everything

**Session:** S2 (golden set v1 + ExactMatchScorer)

**What:** First end-to-end eval. 10 hand-written question/expected pairs,
asked to a local Ollama (`llama3.2`, temperature=0), scored with strict
exact-match: `output == expected`, no normalization.

**Result:** Almost every item failed.

**Why it matters:** This is the lesson exact match is in the suite to
teach, not a bug to fix. The model knows the answer — it just wraps it
in conversational prose:

| Expected   | Got                                       |
| ---------- | ----------------------------------------- |
| `Paris`    | `The capital of France is Paris.`         |
| `Ag`       | `The chemical symbol for silver is Ag.`   |
| `8`        | `There are 8 planets in our solar system.`|

A scoring function that calls all three of those *wrong* is not a useful
scoring function for open-ended LLM output. It's the honest baseline
that motivates everything that follows: semantic similarity, then
LLM-as-judge.

A subtler lesson: this is also why "make the prompt more strict" doesn't
fully fix it. You can squeeze a model toward bare answers, but at
inference time, with non-zero temperature, on questions outside your
prompt-engineered sweet spot, the prose creeps back. The eval framework
has to tolerate it. The model isn't the variable to tighten; the scorer
is.

---

## Finding 2 — A textbook paraphrase scores 0.725, below a "reasonable" 0.75 threshold

**Session:** S3 (SemanticScorer with `all-MiniLM-L6-v2`)

**What:** Wrote the unit test for the new semantic scorer. The whole
*reason* this scorer exists is to pass prose-wrapped right answers.
First test pair was the canonical case:

- output:   `"The capital of France is Paris."`
- expected: `"Paris"`
- threshold: `0.75` (a defensible-sounding default)

**Result:** Cosine similarity = **0.725**. Verdict: **fail.**

**Why it matters:** Two things at once.

1. **Embedding distance is not human "similarity."** A sentence and a
   single word that mean the same thing intuitively are still a sentence
   and a single word — different lengths, different surrounding tokens,
   different contextual embeddings. `all-MiniLM-L6-v2` puts them around
   0.72, not 0.95. If you pick a threshold by gut feel ("0.8 sounds
   strict but fair"), you will reject right answers.

2. **Thresholds are not universal — they're a function of (model,
   dataset).** A threshold tuned on long-form Q&A will be wrong for
   short factual answers, and vice versa. The only honest way to pick
   one is to run the scorer on your real golden set and look at the
   distribution. We left the default at 0.75 deliberately so the live
   eval would generate the real data.

**Implication for the unit test:** the test should assert *behavior*
(paraphrase scores meaningfully higher than unrelated text), not the
specific verdict at a default threshold. The verdict at a given
threshold is a downstream tuning question, not a property of the
scorer.

---

## Reference — how the two scorers work (plain language)

Kept here for the writeup so the explanation isn't scattered across
chat. Two scorers in the suite right now; a third (LLM-as-judge) joins
in S4.

### Exact match

Literally `output == expected`. No normalization — case, whitespace, and
punctuation all count. On open-ended LLM output it fails almost always,
because the model wraps the right answer in prose (`"The capital of
France is Paris."` vs `"Paris"`). It's in the suite as the **honest
baseline** — the wall of red is the motivation for everything else.

### Semantic similarity — built up in four stages

**Stage 1 — Text into numbers (embeddings).** A computer can't compare
meanings, only numbers. So the first move is to turn each piece of text
into a list of numbers — an *embedding*. Our model
(`all-MiniLM-L6-v2`) outputs **384 numbers per input**, regardless of
input length. They aren't human-labeled features; they're coordinates.

**Stage 2 — A meaningful "address space."** The embedding model has
been trained so texts with similar meanings end up near each other in
that 384-dimensional space. `"Paris"` and `"The capital of France is
Paris."` are pulled close. `"Paris"` and `"Water boils at 100°C."` are
pushed apart. The model doesn't *know* what Paris is — it has just seen
enough text that the surrounding-word patterns line up.

**Stage 3 — Measuring "close" with cosine similarity.** Imagine each
embedding as an arrow pointing out from the origin. Cosine similarity
measures the **angle** between two arrows (not the distance). Same
direction → 1.0. Perpendicular → 0.0. Opposite → −1.0. Length doesn't
matter, which is why a single word and a full sentence can still come
out highly similar.

**Stage 4 — From score to verdict.** Pick a threshold; pass if cosine ≥
threshold. We start at 0.75. Move it down → permissive (lets prose
through but also lets topically-related-but-wrong answers through).
Move it up → strict (collapses back toward exact match). There is no
"right" threshold in the abstract — only right for *this model + this
dataset*.

### Why we run both, not just the better one

Each scorer has a failure mode. Exact match is too strict. Semantic
similarity has its own blind spots: a wrong-but-topically-similar
answer can sneak past the threshold, and a right-but-tersely-phrased
answer can fall below it. Running both makes those blind spots visible
as **disagreements** — which is the central artifact of the project.
The judge in S4 is the third opinion that breaks ties.

---

## Finding 3 — Semantic similarity is not measuring correctness

**Session:** S3 (after human-eval round 1, see `reports/human_eval_v1.md`)

**What:** Hand-graded all 10 items from the V1 live run. As a human
grader, I would pass 9 of 10 items — only `procedural_001` is
genuinely wrong (the model hallucinated a non-existent pytest flag).

**Result:** Eight disagreements where semantic said FAIL but a human
would PASS. Zero disagreements the other direction.

**The naive fix is to lower the threshold. It doesn't work. Here's why.**

The lowest "right answer" score and the only "wrong answer" score sit
inside each other:

```
factual_002 (right answer)     0.583
procedural_001 (wrong answer)  0.579     <- only 0.004 apart
factual_003 (right answer)     0.569
definition_001 (right answer)  0.551
```

Any threshold that passes the first row also passes the second. You'd
trade false negatives for false positives 1-for-1. The threshold isn't
the variable that fixes this — there is no threshold that fixes this.

And then there's `factual_004`. Expected `"8"`. Model returned a
bulleted list of all eight planets, plus a footnote about Pluto. Human
verdict: PASS. Cosine: **0.194**. The factual content is identical;
the *shape* of the text isn't. Embedding distance can't see past the
shape. No reasonable threshold rescues this item either.

**Why it matters:**

> Semantic similarity is not measuring correctness. It is measuring
> textual proximity. The two are correlated but they are not the same.
> A right answer in an unusual format scores low. A wrong answer with
> the right vocabulary scores middling. One number cannot answer both
> questions.

This is the empirical case for the LLM-as-judge scorer in S4. We need
a scorer that asks "does this answer the question?" rather than "do
these two strings look like each other?"

**Decision on the threshold:** leaving it at 0.75. Tuning it down to
inflate the pass rate would obscure the real finding. The honest
verdict from S3 is "4/10, and the failure mode is structural, not a
threshold problem."

---
