"""Shared xfail-as-contract helper.

Both `test_calibration.py` and `test_eval_pipeline.py` use the same
pattern: parametrize over a set of ids, mark known-disagreement ids as
`xfail(strict=True)` with a plain-English `reason=` describing the
finding. `strict=True` is the load-bearing flag — if a documented
disagreement silently goes away, the suite breaks (XPASS), so findings
can't be lost without re-investigation.

The disagreement dicts themselves stay inline in each test file because
the two tests encode different invariants:

  * calibration  : scorer verdict vs HUMAN verdict on a frozen output.
  * eval pipeline: scorer says PASS on a live model answer.
"""

from __future__ import annotations

from typing import Iterable

import pytest


def with_xfail(values: Iterable, known_disagreements: dict) -> list:
    """Wrap each value as a pytest.param, xfail-strict if in the dict.

    `known_disagreements` maps value -> plain-English reason. Values not
    in the dict are emitted as plain params (expected to pass).
    """
    params = []
    for v in values:
        if v in known_disagreements:
            params.append(
                pytest.param(
                    v,
                    marks=pytest.mark.xfail(
                        strict=True,
                        reason=known_disagreements[v],
                    ),
                )
            )
        else:
            params.append(pytest.param(v))
    return params
