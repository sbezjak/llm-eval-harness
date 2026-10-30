"""LLM-as-judge: a second model grades the first model's answer.

Plain-English version of what this file does, end to end:

1. Take the question, the expected answer, and the model's actual
   answer. Drop them into a grading-instructions template (see
   `RUBRIC_PROMPT`).
2. Send that filled-in prompt to an LLM (a *separate* call from
   whatever produced the answer being graded).
3. The LLM replies with a small JSON blob containing two integer
   scores (correctness 0-10, relevance 0-10), a verbatim `evidence_span`
   quoted from the answer, and a `contradicts_expected` flag.
4. Parse the JSON. Substring-check the `evidence_span` against the
   answer - a judge that cannot quote what it read (or quotes something
   that isn't there) is not trusted, it raises. Then average the two
   scores, divide by 10 to get a score in [0, 1], compare against a
   threshold, return PASS or FAIL.

Why bother. The simpler scorers (exact match, BLEU, ROUGE, semantic
similarity) all compare *text shapes*. They can be fooled by:

- A right answer phrased very differently from the expected (e.g. a
  bulleted list of all 8 planets when the expected was just "8" -
  exact match and BLEU/ROUGE see no overlap; embeddings see different
  shape).
- A wrong answer that uses the right vocabulary (semantic similarity
  passes it).

A judge LLM reads both texts and can sometimes recognize "yes, this
answers the question" or "no, this is plausible-sounding nonsense"
where pure-text comparison can't.

Important caveat. When the judge and the answering model are the same
weights (the case in this project - both are llama3.2), the judge has
a blind spot for its own hallucinations. It produced the same
nonsense; it can't tell it apart from a real answer. Surfacing this
"self-grading bias" is one of the things this harness is here to
demonstrate - see `tests/test_calibration.py` and the
`procedural_001` item in `data/human_labels.yaml`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from eval_harness.providers.ollama import OllamaProvider
from eval_harness.scorers.base import Scorer, ScoreResult

DEFAULT_THRESHOLD = 0.7
DEFAULT_MODEL = "llama3.2"

# Characters a judge model swaps in when it *did* read the answer but did not
# copy it byte-for-byte: every flavour of single/double quote, backtick, prime.
# See _normalize_for_match.
_QUOTE_CHARS = "\"'`´‘’‚‛“”„‟′″"
_WHITESPACE_RE = re.compile(r"\s+")
# A markdown code-fence line (```, ```bash). Formatting, not words.
_FENCE_LINE_RE = re.compile(r"^[ \t]*```[\w+-]*[ \t]*$", re.MULTILINE)


def _normalize_for_match(text: str) -> str:
    """Fold the *cosmetic* differences a judge introduces when quoting the
    answer, without touching word content, so the grounding substring check
    stays robust to them:

    - all quote characters (straight, curly, backtick, prime) are removed -
      weak models freely swap ``"AI"`` for ``’AI’``;
    - code-fence lines (```` ``` ````, ```` ```bash ````) are dropped - a judge
      quoting across a code block skips the fence (observed live 2026-10-30:
      ``option:\\n\\n```bash\\npytest`` quoted as ``option:pytest``);
    - all whitespace (incl. newlines) is removed - weak models re-flow
      ``:\\n\\n1.`` to ``:\\n1.``, or drop the line break entirely.

    Deliberately narrow: it does NOT lowercase, stem, or drop words, so a
    genuinely fabricated span (different words, invented tokens like ``pysto``,
    a quote of the question instead of the answer) still fails the check. This
    is a tier-3 mechanical tolerance, not a fuzzy match - it keeps the guard's
    teeth while forgiving punctuation-and-whitespace noise the model cannot be
    trusted (finding: stating "copy exactly" is inert) to avoid.
    """
    unfenced = _FENCE_LINE_RE.sub("", text)
    stripped = unfenced.translate({ord(c): None for c in _QUOTE_CHARS})
    return _WHITESPACE_RE.sub("", stripped)


# The grading instructions sent to the judge. See LLMJudgeScorer's
# docstring for why the rubric looks the way it does.
RUBRIC_PROMPT = """You are a strict but fair grader for a question-answering system.

You will be given:
- A QUESTION
- An EXPECTED answer (a reference - the model's answer does NOT need to
  match it word-for-word to be correct)
- The MODEL'S answer to grade

Grade the model's answer on two dimensions, each 0-10:
- correctness: does the answer correctly answer the question?
  10 = fully correct (even if phrased differently from the expected answer)
   5 = partially correct, or correct but with significant inaccuracies
   0 = wrong, contradicts the expected answer, or hallucinated
- relevance: is the answer on-topic and responsive to the question?
  10 = directly addresses the question
   5 = partially on-topic, drifts
   0 = off-topic or refuses to answer

Respond with ONLY a JSON object, no other text:
{{"evidence_span": "<Copy, word for word, the span of the MODEL'S ANSWER that most determines your correctness score. Copy it exactly so it can be checked against the answer verbatim. Quote the MODEL'S ANSWER itself - NOT the EXPECTED answer, NOT the question, and NOT these instructions. If the answer is wrong, quote the wrong part of the answer, not the correct version.>",
 "contradicts_expected": <true if the model's answer asserts something the EXPECTED answer contradicts, else false>,
 "correctness": <int>,
 "relevance": <int>}}

QUESTION: {question}
EXPECTED: {expected}
MODEL'S ANSWER: {output}"""


class JudgeParseError(RuntimeError):
    """Raised when the judge's response cannot be parsed into scores.

    A malformed judge response is an infrastructure problem (the judge
    model isn't following the rubric format), not "the model's answer
    was bad." Silently coercing to a 0 score would hide bugs in the
    judge prompt or the parsing logic.
    """


def _extract_fields(data: dict) -> tuple[int, int, str, bool]:
    """Pull (correctness, relevance, evidence_span, contradicts_expected) from a
    parsed judge dict. Raises KeyError/TypeError/ValueError to the caller if the
    two scores are missing or non-integer; the grounding fields default softly
    (an absent span is caught later by the substring check in score())."""
    return (
        int(data["correctness"]),
        int(data["relevance"]),
        str(data.get("evidence_span", "")),
        bool(data.get("contradicts_expected", False)),
    )


def _parse_judge_response(raw: str) -> tuple[int, int, str, bool]:
    """Extract (correctness, relevance, evidence_span, contradicts_expected).

    Strategy: try strict JSON first. If the model wrapped the JSON in
    prose or markdown, try to find the first {...} block. As a last
    resort, regex-extract the two integer scores (no grounding fields
    survive that path - an empty span then fails the substring check in
    score(), which is the correct outcome: an unparseable rationale is
    not a grounded one). If even that fails, raise JudgeParseError with
    the raw text - better than silently returning a 0.
    """
    try:
        return _extract_fields(json.loads(raw))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    json_blob = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_blob:
        try:
            return _extract_fields(json.loads(json_blob.group(0)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    correctness_match = re.search(r'"?correctness"?\s*[:=]\s*(\d+)', raw, re.IGNORECASE)
    relevance_match = re.search(r'"?relevance"?\s*[:=]\s*(\d+)', raw, re.IGNORECASE)
    if correctness_match and relevance_match:
        return (
            int(correctness_match.group(1)),
            int(relevance_match.group(1)),
            "",
            False,
        )

    raise JudgeParseError(f"Could not parse judge response: {raw!r}")


class LLMJudgeScorer(Scorer):
    """Calls a grading LLM and turns its reply into a PASS/FAIL.

    Three rubric design choices are worth knowing about, since they
    explain why `RUBRIC_PROMPT` looks the way it does:

    - **Reference is a hint, not a string to match.** The judge is told
      the expected answer is a reference - the model's answer can be
      phrased completely differently and still be correct. This is what
      lets the judge pass right-but-different-shape answers (a bulleted
      list when the expected was a single number, etc).
    - **Two scores, not one.** Asking for *correctness* and *relevance*
      separately keeps two distinct failure modes from being smeared
      into a single number: a right-but-rambling answer scores high on
      correctness and low on relevance; a confident hallucination
      scores low on correctness and high on relevance. A single
      "quality" score loses that signal.
    - **Grounded evidence, not free-text reasoning.** The rubric asks
      the judge to quote, verbatim, the `evidence_span` of the answer it
      based its correctness call on *before* writing the number - and
      `score()` substring-checks that quote against the answer in code.
      A free-text "reasoning" field is unverifiable: the judge can
      fabricate a justification for any number. Forcing a quote the code
      can check turns "the judge read the answer" from a hope into an
      assertion - a fabricated or un-grounded rationale raises rather
      than passing silently. It does NOT check that the score follows
      from the quote: on procedural_001 the judge quoted the plausible
      `-m` lines, skipped the fake flag, and still gave 9/10. (This is the shape-blind grounding guard;
      we never tell the judge *where* the failure is, only that it must
      quote what it read.) The evidence-before-score ordering also keeps
      the older benefit: a model asked for the score first tends to
      commit to a number and rationalize it.

    Construction.

    - `threshold` - pass/fail cutoff on the [0, 1] combined score.
    - `judge_fn` - async callable that takes a prompt and returns the
      judge's raw text. Defaults to a fresh `OllamaProvider` at
      temperature 0 (judge wants determinism). Tests inject a fake
      `judge_fn` to avoid touching HTTP at all.
    - `model` - the Ollama model name to use when `judge_fn` is the
      default.
    """

    name = "judge"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        judge_fn: Callable[[str], Awaitable[str]] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.model = model
        if judge_fn is None:
            provider = OllamaProvider(model=model, timeout=180.0, temperature=0.0)
            judge_fn = provider.generate
        self._judge_fn = judge_fn

    def build_prompt(self, question: str, output: str, expected: str) -> str:
        """Render the rubric prompt the judge will see. Public so tests and
        debug paths can inspect what was actually sent without scoring."""
        return RUBRIC_PROMPT.format(question=question, expected=expected, output=output)

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        prompt = self.build_prompt(question, output, expected)
        raw = await self._judge_fn(prompt)
        correctness, relevance, evidence_span, contradicts_expected = _parse_judge_response(raw)

        # Clamp into 0-10 in case the judge invented an out-of-range score.
        correctness = max(0, min(10, correctness))
        relevance = max(0, min(10, relevance))

        # Grounding guard. The judge must quote, verbatim, the span of the
        # answer it based its correctness call on, and that quote must really
        # be in the answer. A fabricated rationale (a span that isn't in the
        # answer) or an un-grounded verdict (no span at all) is not trusted -
        # it is the "I read it" lie. This is a substring check in *code*, not a
        # request the judge can talk its way past: the judge cannot claim to
        # have read the answer it cannot quote. See the design note in the
        # class docstring; validated shape-blind before rollout.
        span = evidence_span.strip()
        # Normalized too: a span of only quote marks or a fence line folds to
        # "", and "" is in every answer.
        if not _normalize_for_match(span):
            raise JudgeParseError(
                "judge returned no evidence_span - an un-grounded verdict is not trusted"
            )
        if _normalize_for_match(span) not in _normalize_for_match(output):
            raise JudgeParseError(
                f"judge cited a span not in the answer (fabricated rationale): {span!r}"
            )

        # A judge that admits the answer contradicts the reference cannot also
        # pass it on correctness. Cheap tier-3 gate on top of the grounding.
        if contradicts_expected:
            correctness = min(correctness, 2)

        score = (correctness + relevance) / 20.0
        passed = score >= self.threshold
        reason = (
            f"correctness={correctness}/10 relevance={relevance}/10 "
            f"score={score:.3f} {'>=' if passed else '<'} threshold={self.threshold} "
            f"| grounded on: {span!r}"
            + (" | judge flagged contradiction with EXPECTED" if contradicts_expected else "")
        )
        return ScoreResult(passed=passed, score=score, reason=reason)
