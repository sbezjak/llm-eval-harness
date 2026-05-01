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

## Design decision — async-only scorers vs. dual sync+async (the production pattern)

**Session:** S4 (when adding LLM-as-judge, which needs an HTTP call inside `score()`)

**The fork.** The first two scorers (`ExactMatchScorer`, `SemanticScorer`) had no I/O at score time, so the base class's `score()` method was sync. The judge does have I/O — every score call is an HTTP request to Ollama. Three ways to resolve:

- **A.** Make `Scorer.score()` async across the board. Refactor existing scorers (trivial — they just don't await anything) and the handful of test call sites.
- **B.** Keep `score()` sync; judge uses a sync `httpx.Client` internally. The async `OllamaProvider` abstraction can't be reused — parallel sync/async transport code.
- **C.** Inject a `Callable[[str], str]` into the judge. Same downside as B, hidden behind a callable.

**What production frameworks do.** None of the above, exactly. They use a **dual interface**: every scorer exposes both a sync and an async method.

- DeepEval: `measure()` and `a_measure()`
- Ragas: `score()` and `ascore()`
- LangChain evaluators: `evaluate_strings()` and `aevaluate_strings()`
- Braintrust autoevals: scorer is sync at the call boundary, takes an LLM client at construction; the client encapsulates the transport

The reason is **async contagion.** Many scorers have no I/O at all — exact match, regex, BLEU, ROUGE. If you make every scorer async, every consumer must be async too. Someone calling your scorer from a Jupyter notebook or a sync CLI shouldn't need an event loop. Conversely, async-only scorers can't be called from a sync context without `asyncio.run()`, which **breaks** if you're already inside an event loop (which is the situation in any `async def` test).

So the strictly-most-production-like move would be: scorers expose both `score()` and `ascore()`, and `OllamaProvider` gains a sync `generate_sync()` alongside its async `generate()`.

**What we did and why.** Option A — async everywhere. *Not* because it's best practice, but because for this project's scope it's the right trade-off:

1. **One runtime context.** Pytest with `asyncio_mode = "auto"`. Every test is already async. Dual interface solves a problem this harness doesn't have.
2. **Consistency with existing choices.** `OllamaProvider` is async-only. The pipeline test is async. Going async-everywhere extends an existing decision instead of introducing a second pattern.
3. **Test-count budget.** The plan caps this project at 30–35 tests. Doubling the scorer surface area (sync + async versions, plus tests for both code paths) eats budget that's better spent on the bias test, calibration test, and dataset expansion.

**Why this is worth writing down.** Async-only is fine for a focused harness; it would be wrong for a library. The distinction matters: production frameworks have dual interfaces because they serve thousands of users in different runtime contexts, not because async-only is "wrong." Calling out the trade-off explicitly is what separates a deliberate scope choice from an oversight.

---

## Reference — designing an LLM-as-judge scorer

Kept here for the writeup. The four design decisions that actually shape the judge.

**1. Reference-based vs reference-free vs hybrid.**

- *Reference-based:* judge sees the expected answer and asks "is the output consistent with this?" Becomes a smarter `SemanticScorer` — sees past wording and format. Catches the planets-list case (the model's bulleted list of all 8 planets is correct, but `factual_004` expected `"8"`).
- *Reference-free:* judge sees only the question and the model's answer, and uses its own world knowledge. Maximally exposes self-grading bias.
- *Hybrid:* judge sees the expected answer as a *reference*, but is told the model's answer doesn't have to match it verbatim — it just has to be a correct answer to the question.

We chose **hybrid.** Reference-free overestimates llama3.2's knowledge. Reference-based collapses into "fancy semantic similarity" and loses the ability to handle right-but-different-shape answers.

**2. How many rubric dimensions.**

Small local models get confused with 5+ dimensions and start hallucinating scores or ignoring the rubric. Big models tolerate many. For llama3.2 the right number is **two:**

- *correctness* — does the answer answer the question, factually?
- *relevance* — is it on-topic and responsive?

Splitting these matters for the `procedural_001`-style failure: the answer *looks* relevant (mentions pytest, mentions flags) but is factually wrong. A single combined "quality" score smears the two failure modes together and you lose that signal.

**3. Output format.**

JSON is ideal for parsing but small local models produce malformed JSON ~10–30% of the time, especially with multi-key schemas. Two robustness moves:

- Force a minimal schema: `{"reasoning": "...", "correctness": <int>, "relevance": <int>}`. Three keys, two of them ints.
- Have a parser fallback. Try `json.loads` first; on failure, regex-extract `correctness:\s*(\d+)` and `relevance:\s*(\d+)` from the raw text. If *that* also fails, raise — don't silently coerce to 0. A malformed judge response is an infrastructure problem, not "the answer was bad," and silent coercion would hide bugs.

**4. Reasoning before or after the score.**

Counterintuitively: **reasoning first, score second.** If you ask for the score first, the model commits to a number and post-hoc rationalizes it. Reasoning-first forces the model to think before scoring. Same mechanism as chain-of-thought.

**Why we chose two scores instead of one combined score.**

Final pass/fail is computed as `(correctness + relevance) / 20 ≥ threshold`. We could have asked the judge for a single 0–10 "overall quality" instead. The two-dimension design is worth the extra token cost because:

- The two dimensions fail independently. A right-but-rambling answer scores high on correctness, low on relevance. A confident hallucination scores low on correctness, high on relevance. One number can't distinguish those.
- When we look at disagreements between scorers, having `correctness` separately tells us *why* the judge disagreed with semantic — was it about facts, or about focus?

**Expected pathologies (to look for in the live run, not assumed).**

- *Self-grading bias.* Judge is the same llama3.2 that produced the answer. We expect it to be soft on its own outputs — recognize its own phrasings as "answering the question," and possibly accept its own confident hallucinations. This is the centerpiece finding to look for.
- *Score clustering.* Small models cluster on round numbers (5, 7, 10) and effectively give you a 3–4-bucket scale instead of an 11-bucket one. If we see this, fine — note it; it's an honest finding, not a bug. If it's pathological we can fall back to a 1–5 scale.

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

## Finding 4 — Self-grading bias is real (and it sits exactly at the threshold)

**Session:** S4 (LLM-as-judge against the V1 golden set)

**What:** Same setup as before, with the new `LLMJudgeScorer` (hybrid
rubric, correctness + relevance, threshold 0.7) added to the matrix.
The judge call is the same `llama3.2` that produced the answers —
self-grading by design, on the assumption that a different judge model
is project 5's job, not this one's.

**Headline result:** The judge passed `procedural_001` — the only item
where the model is genuinely wrong (a hallucinated `--junit-xml-filter`
flag that does not exist in pytest). The judge gave it correctness 8/10,
relevance 6/10, score **0.700** — passing exactly at the threshold.

The judge's reasoning text:

> "The model provided multiple correct ways to run the test, but failed
> to directly answer the question. The model's answer is partially
> relevant as it does provide a solution, but could be more concise..."

There is no "multiple correct way to run the test" — every flag in the
output was invented. The same model that hallucinated them is grading
them as plausible because it doesn't know either. This is self-grading
bias in textbook form: a judge cannot detect the hallucinations of a
model that shares its weights.

**Per-item judge scores:**

```
procedural_001  0.700  PASS  ← model is wrong
reasoning_001   0.850
reasoning_003   0.850
factual_003     0.900
definition_002  0.950
factual_001     1.000
factual_002     1.000
factual_004     1.000  ← rescued from semantic 0.194 (the planets list)
definition_001  1.000
reasoning_002   1.000
```

**The nuance — judge as a noisy detector, not a blind one.** Note that
procedural_001 *is* the lowest-scoring item by the judge. The judge
pushed the only wrong answer to the bottom of the distribution. It just
landed at the threshold rather than below it. So the judge has *some*
signal — the ranking is right — but the absolute correctness score is
way too generous (8/10 for a fully fabricated answer).

**Why it matters.**

1. **The empirical case for cross-model judging.** This is the data
   point that justifies using a different model as judge in project 5.
   "Self-grading bias matters in practice" is no longer an assertion —
   it's a row in the report.
2. **Threshold tuning is a band-aid here.** Raising the threshold to
   0.71 would make procedural_001 fail, but it would also start
   failing items the judge ranked higher with weaker confidence. You
   trade a false positive on a known-wrong item for false negatives
   on right items. The fix is not the threshold; the fix is a
   different judge.
3. **factual_004 is the win the judge was added for.** Semantic scored
   the planets-list answer at 0.194 (it didn't recognize "list of 8
   things" as equivalent to "8"). The judge scored it 1.000. That
   single rescue is the empirical justification for the judge in the
   suite — it sees past textual shape in a way semantic structurally
   cannot.

---

## Finding 5 — The judge's reasoning and score decouple, with a recurring rationalization tic

**Session:** S4 (V2 report, three items)

**What:** While reading the V2 judge reasoning text, three items
showed the same wrong rationalization — a fixed phrase the judge fell
back on whether or not it applied:

| Item | Model's answer (abridged) | Judge's reasoning |
|---|---|---|
| factual_003 | "Lima is the capital of Peru, and it is located in **South America**." | "correctly identifies Lima as being in South America, but **does not directly address the question** of which continent it is located on." |
| reasoning_001 | Long, detailed explanation of cold/warm-blooded animals | "clear explanation… However, **it did not directly address the question** in its initial response." |
| procedural_001 | Hallucinated pytest flags | "The model provided multiple correct ways to run the test, but **failed to directly answer the question**." |

In each case the model literally addressed the question. The judge
generated the same "did not directly address the question" rationalization
three times across unrelated items — a stylistic tic, not a real
critique.

**The deeper observation:** the *scores* on those items were roughly
correct (0.85, 0.85, 0.70). The *reasoning text* was wrong. The two
parts of the judge's output decoupled.

**Why it matters.**

1. **Trust the score, treat the reasoning as a hint.** This is a
   surprising finding — the obvious assumption is that reasoning-first
   chain-of-thought makes the score *more* trustworthy because the
   model "thought it through." On a small local model, what we see
   instead is: the score reflects something close to the right
   intuition, and the prose is post-hoc filler that follows
   trained-pattern templates ("does not directly address the
   question" is a phrase that appears in *evaluation-style* training
   data; the model produces it whether or not it applies).
2. **Implications for production use.** If you wire an LLM judge into
   a CI gate, gate on the score — not on parsed reasoning text, not on
   sentiment of the prose. The score is the signal; the prose is
   decoration.
3. **This is a small-model artifact and we should expect it to shrink
   on bigger judges.** A claim to test in project 5 (cross-model
   benchmarking): does the same rubric prompt produce coherent,
   item-specific reasoning when judged by GPT-4o or Claude Sonnet?
   The hypothesis is yes — that small-model judges hallucinate
   *reasoning* the way they hallucinate *answers*, just in a different
   register.

---
