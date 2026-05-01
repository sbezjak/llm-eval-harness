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
