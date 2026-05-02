# Scoring trade-offs

This doc collects the deliberate trade-offs in this harness in one place,
plus two reference tables a QA reader will want before reading
`article.md`'s findings:

1. **QA translation table** — every concept in this repo mapped to the
   equivalent in traditional software QA, so you can read the rest of
   the project in vocabulary you already have.
2. **Failure-mode matrix** — which scorer catches which kind of LLM
   failure, and where each scorer is structurally blind. This is the
   evidence behind the project's thesis: *no single scorer is enough.*

The trade-offs themselves follow.

---

## 1. QA translation table

LLM evaluation is unfamiliar vocabulary for a lot of QA engineers, but
most of the building blocks have direct analogues in traditional test
practice. If you read the rest of this repo with this table in mind,
nothing in it should feel exotic.

| LLM-eval concept (this repo) | What it is in traditional QA | Why the LLM version is different |
|---|---|---|
| `ExactMatchScorer` | A string-equality assertion (`assert output == expected`) | Output is non-deterministic prose, so strict equality fails on right answers — see Finding 1 |
| `SemanticScorer` (cosine ≥ 0.75) | A fuzzy/tolerance assertion (`abs(a - b) < eps`, regex match, percent-similar diff) | The "tolerance" is a learned embedding distance, not a numeric epsilon — and the threshold is a function of (model, dataset), not a universal constant (Finding 2) |
| `LLMJudgeScorer` | Automated manual review — a reviewer applying a rubric, but driven by a second model | The reviewer can have systematic bias (self-grading bias, length bias, score-clustering) that a human reviewer doesn't have in the same shape — Findings 4, 8, 9 |
| `data/golden_set.yaml` | A test fixture / test-data file | The "expected" field isn't the only acceptable answer; it's a reference, and the scorer decides whether the actual output is *equivalent enough* |
| `data/human_labels.yaml` | A frozen baseline / golden master for regression testing | The thing being baselined is the *model's output text*, paired with a *human verdict* — both are needed because the model drifts and the scorer's job is to agree with the human, not the model |
| `tests/test_calibration.py` | Tests on the test-tooling itself ("test the linter") | Calibration measures *agreement between the scorer and a human*, which is the only honest measure of scorer quality — see Finding 6 |
| `xfail(strict=True, reason=...)` | Known-issue tracking (a `@pytest.mark.skip("bug #123")` that is *expected* to fail and breaks the build if it stops failing) | Each xfail encodes a *finding* — a documented structural limitation of a scorer — not a bug. If the limitation silently disappears the suite breaks, forcing us to update the writeup |
| `tests/test_bias.py` (paired prompts) | Metamorphic testing — change one input variable, assert the output changes / doesn't change in a known way | The relation being asserted is "factual content invariant under name swap." The test is a sanity check, not a bias-benchmark replacement |
| `tests/test_judge_variance.py` | Flakiness / stability test (run N times, assert variance bounds) | Catches both flakiness *and* stuck-at-threshold determinism (Finding 8) — the latter is more dangerous because it doesn't show up as a flaky red |
| `@pytest.mark.ollama` vs `@pytest.mark.mocked` | Integration tests vs unit tests with a mocked dependency | The "integration" target is a model whose outputs can drift between runs — the marker split is also a *cost* split, since live runs cost minutes and tokens |
| `eval_harness/providers/` | The boundary where you mock external services in a test (HTTP client, DB driver) | `respx` mocks at the HTTP layer just like in any FastAPI/httpx project — nothing LLM-specific about the technique |

The takeaway: LLM evaluation is software testing with two extra
constraints — (1) the output is non-deterministic prose, so equality
won't carry you, and (2) the *scorer itself* is fallible and has to be
calibrated against humans. Everything else is pytest.

---

## 2. Failure-mode matrix

Each row is a category of LLM output failure (or a thing that *looks*
like a failure but isn't). Each cell answers: *does this scorer correctly
verdict this case?* Sourced from Findings 1–9 in `article.md`; the
finding number is in parentheses where the cell is a documented result
of this project rather than a general claim.

Legend: ✅ correct verdict · ❌ wrong verdict · ⚠️ correct verdict but
fragile / threshold-dependent · — not applicable.

| Failure mode (illustrative example) | Exact match | Semantic | LLM-as-judge |
|---|---|---|---|
| **Right answer wrapped in prose** (`"The capital of France is Paris."` vs `"Paris"`) | ❌ (Finding 1) | ⚠️ borderline — 0.725 below 0.75 (Finding 2) | ✅ |
| **Right answer in a different shape** (bulleted list of 8 planets vs `"8"`) | ❌ | ❌ score 0.194 (Finding 3) | ✅ rescues to 1.000 (Finding 4) |
| **Confident hallucination** (invented pytest flag) | — (n/a, exact match wouldn't pass it either) | ⚠️ topically similar, may pass | ❌ self-grading bias passes it at exactly 0.700 (Findings 4, 8) |
| **Stuck-at-threshold determinism on the hallucination** | — | — | ❌ scores 0.700 every run; *cannot* be averaged out (Finding 8) |
| **Demographic substitution drift** (David vs Priya same prompt) | — | ✅ catches structural drift below 0.7 cosine (Finding 7) | not used in bias test |
| **Length-only variation on a correct answer** (short vs long correct) | varies | varies with threshold | ✅ no detectable bias on llama3.2 (Finding 9) |
| **Reasoning text vs final score on the same item** | — | — | ⚠️ score is right, prose is post-hoc filler (Finding 5) |
| **Same-content paraphrase, different vocabulary** | ❌ | ✅ (the case it was designed for) | ✅ |
| **Wrong but topically similar answer** (right vocabulary, wrong fact) | ✅ (will fail it for the wrong reason) | ❌ middling cosine, may pass | depends on judge knowledge of the fact |
| **Refusal / "I don't know"** (not yet in golden set) | ❌ unless expected is also a refusal | ❌ low cosine to expected | ⚠️ judge may rate as "relevant 0, correct 0" — untested |

**What this matrix proves.** No column is all-✅. Every scorer has at
least one ❌ that another scorer covers. The disagreement test
(`test_eval_pipeline.py::test_scorers_agree_on_verdict`) is what
surfaces those ❌ cells in practice — when scorers disagree, *one of
them is in a ❌ row*, and you go look at the per-item output to see
which.

**What this matrix does not prove.** That running all three is
sufficient. The bottom row (refusals) is a known gap: nothing in the
current suite handles "the model correctly declined to answer" well.
That's a future-finding row, not a current-coverage claim.

---

## 3. Trade-offs written down

The conventions in this repo aren't all best-practice; some are
deliberate scope cuts for a learning project. They're listed here so a
reader doesn't have to guess which is which.

### 3.1 Async-only scorers (vs production's dual sync+async)

The `Scorer.score()` contract is async across all three scorers, even
the two that don't do I/O (exact match, semantic). Production
frameworks (DeepEval `measure`/`a_measure`, Ragas `score`/`ascore`,
LangChain `evaluate_strings`/`aevaluate_strings`) expose both
interfaces because they serve consumers in many runtime contexts —
notebooks, sync CLIs, async services — and async-only forces async on
all of them ("async contagion").

We picked async-only because this harness has *one* runtime context
(pytest with `asyncio_mode = "auto"`). Dual-interface would double the
scorer surface area and the test count for a problem this project
doesn't have. The full reasoning is in `article.md` ("Design decision —
async-only scorers").

**When this would be wrong.** If this harness ever became a library
imported by sync code, the dual interface would be the right move.

### 3.2 Frozen calibration corpus (`data/human_labels.yaml`)

`tests/test_calibration.py` doesn't re-run the model — it scores
*frozen Ollama outputs* committed to `data/human_labels.yaml` against
human verdicts from `reports/human_eval_v2.md`. Two reasons:

- **Separates two regressions.** A test that re-runs the model on
  every invocation conflates "did the scorer regress?" with "did the
  model drift?" Freezing the corpus means a calibration failure is
  unambiguously a scorer change, not noise from `temperature=0` not
  being deterministic in practice.
- **Human verdicts are tied to specific text.** The label "PASS" was
  assigned to a *specific output*. Re-running could produce
  near-identical text the human would *also* pass, but you'd have no
  way to know without re-grading.

**When this would be wrong.** When you want to detect model drift
itself — then you *want* the live re-run, and you compare today's
output against the frozen one as a separate test.

**Operational note.** Re-grading is a deliberate act, not a
side-effect of running tests. To refresh the corpus you re-run the
live golden-set eval, hand-grade the new outputs into a new
`reports/human_eval_v3.md`, and update `data/human_labels.yaml` from
that.

### 3.3 xfail-as-contract instead of agreement statistics (Cohen's κ)

The standard metric for scorer-vs-human agreement is Cohen's κ. With
one rater and 10 items, κ has a 95% CI of roughly [-0.3, +0.9] — a
number with no decision power. So `test_calibration.py` does not
report κ.

Instead, every (item × scorer) pair runs as a parametrized test, and
the pairs where Findings 1, 3, 4, 7 say the scorer structurally
disagrees with humans are marked
`pytest.mark.xfail(strict=True, reason="<plain-English finding>")`.
The set of xfails *is* the contract:

- agreements stay green
- documented disagreements stay yellow (xfail) with their reason
- if a disagreement ever silently flips to agreement (`XPASS`), the
  suite breaks on purpose — we re-investigate, we don't lose the
  finding

**When this would be wrong.** Once n is in the hundreds and you have
multiple raters, κ becomes informative and you'd report it alongside
xfail-as-contract, not instead of it.

### 3.4 No BLEU/ROUGE in the original scorer set

The original plan listed BLEU/ROUGE in the "Learn" column. We didn't
build them in S2–S5 because Finding 1 already makes the n-gram
failure-mode argument: any n-gram-based score on `"Paris"` vs
`"The capital of France is Paris."` will be near zero, for the same
reason exact match fails. Adding BLEU and ROUGE would have produced
two more rows in the failure-mode matrix that say the same thing as
the exact-match row.

**Why they're back in S6.5.** A reader who comes from the BLEU/ROUGE
world deserves to see the result on this corpus rather than be told
about it. The S6.5 addition runs both scorers across the golden set,
adds a Finding 10 ("BLEU/ROUGE share exact match's n-gram failure
mode"), and adds two columns to the failure-mode matrix above.

**When this would be wrong.** For tasks where n-gram overlap *is* the
right metric (translation, summarization with reference summaries,
generation with strong format constraints) BLEU/ROUGE are the right
default, not a second-class addition.

### 3.5 18-item golden set, not 50 or 200

The plan originally called for 50 hand-written items + 200
LLM-generated ones. We cut both at end of S4. The 10 V1 items already
produced 5 distinct findings (~1 finding per 2 items); 240 more items
of the same shape would add evidence weight without teaching new
concepts. Quality > size for an *eval-harness writeup*. The 200
synthetic items are deferred to S7 as a *separate* dataset that
teaches a *separate* lesson (synthetic blind spots vs hand-written),
not as a backfill.

**When this would be wrong.** For a production eval harness deployed
against a model whose outputs are evaluated for release decisions, n
in the hundreds is the floor, not a stretch goal. Tiny golden sets
are right for *learning what to measure*; production calibration is
how you measure it credibly.
