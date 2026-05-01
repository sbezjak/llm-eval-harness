from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness.dataset import GoldenItem, load_golden_set


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "golden.yaml"
    p.write_text(body)
    return p


def test_loads_real_golden_set():
    items = load_golden_set()
    assert len(items) >= 10
    assert all(isinstance(i, GoldenItem) for i in items)
    ids = [i.id for i in items]
    assert len(set(ids)) == len(ids)
    assert {i.difficulty for i in items} <= {"easy", "medium", "hard"}


def test_rejects_empty_file(tmp_path: Path):
    p = _write(tmp_path, "")
    with pytest.raises(ValueError, match="non-empty list"):
        load_golden_set(p)


def test_rejects_missing_required_field(tmp_path: Path):
    p = _write(
        tmp_path,
        "- id: x\n  question: q\n  expected: e\n  difficulty: easy\n",
    )
    with pytest.raises(ValueError, match="missing fields"):
        load_golden_set(p)


def test_rejects_invalid_difficulty(tmp_path: Path):
    p = _write(
        tmp_path,
        "- id: x\n  question: q\n  expected: e\n  difficulty: trivial\n  category: factual\n",
    )
    with pytest.raises(ValueError, match="invalid difficulty"):
        load_golden_set(p)


def test_rejects_duplicate_ids(tmp_path: Path):
    body = (
        "- id: dup\n  question: q1\n  expected: e1\n  difficulty: easy\n  category: factual\n"
        "- id: dup\n  question: q2\n  expected: e2\n  difficulty: easy\n  category: factual\n"
    )
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="Duplicate item id"):
        load_golden_set(p)
