"""Print, in full, what the guard sees on procedural_001: the answer, the judge's raw reply, and the checks."""
import asyncio, re, yaml
from pathlib import Path
from eval_harness.dataset import load_golden_set
from eval_harness.providers.ollama import OllamaProvider
from eval_harness.scorers.judge import LLMJudgeScorer, _parse_judge_response, _normalize_for_match

def checks(span, output):
    nospace = lambda s: re.sub(r"\s+", "", _normalize_for_match(s))
    print("in answer, current guard:", _normalize_for_match(span) in _normalize_for_match(output))
    print("in answer, all whitespace removed:", nospace(span) in nospace(output))

async def judge(label, question, output, expected):
    print(f"\n===== {label} =====\n--- ANSWER (full) ---\n{output}\n--- end ANSWER ---")
    raw = await OllamaProvider(model="llama3.2", timeout=180.0, temperature=0.0).generate(
        LLMJudgeScorer().build_prompt(question, output, expected))
    print(f"--- JUDGE RAW (full) ---\n{raw}\n--- end JUDGE RAW ---")
    c, r, span, contra = _parse_judge_response(raw)
    print(f"correctness={c} relevance={r} contradicts_expected={contra}\nspan={span!r}")
    checks(span, output)

async def main():
    item = next(i for i in load_golden_set() if i.id == "procedural_001")
    live = await OllamaProvider(model="llama3.2", timeout=180.0, temperature=0.0).generate(item.question)
    await judge("LIVE answer (pipeline + variance tests)", item.question, live, item.expected)
    labels = yaml.safe_load(Path("data/human_labels.yaml").read_text())
    rows = labels if isinstance(labels, list) else labels.get("items", labels)
    row = next(x for x in rows if x["id"] == "procedural_001")
    await judge("FROZEN answer (calibration test)", row["question"], row["frozen_output"], row["expected"])

asyncio.run(main())
