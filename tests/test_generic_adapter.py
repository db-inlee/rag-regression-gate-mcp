"""T3 — Generic adapter equivalence (pins B1/B2/B3 manual checks as tests).

The config-driven Generic components must reproduce the dedicated Allganize
adapter EXACTLY. These pin the 40/40 equivalences verified by hand so any future
edit to app/adapters/generic.py that drifts from the dedicated behavior is caught.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.adapters.allganize import (AllganizeEvalSetProvider, AllganizeGoldMatcher,
                                    allganize_value_present)
from app.adapters.generic import (allganize_eval_provider, allganize_gold_matcher,
                                  make_value_present)

BASELINE = Path("examples/allganize_baseline")


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return [c.model_dump() for c in AllganizeEvalSetProvider().load()]


@pytest.fixture(scope="module")
def run_records() -> dict:
    recs = {}
    for line in (BASELINE / "run.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("type") == "case":
            recs[o["id"]] = o
    return recs


# --- B1: GoldMatcher --------------------------------------------------------- #

def test_generic_goldmatcher_matches_dedicated(cases, run_records):
    dedicated, generic = AllganizeGoldMatcher(), allganize_gold_matcher()
    assert len(cases) == 40
    for c in cases:
        rec = run_records[c["id"]]
        assert dedicated.gold_refs(c) == generic.gold_refs(c)
        assert dedicated.retrieved_refs(rec) == generic.retrieved_refs(rec)


# --- B3: EvalSetProvider ----------------------------------------------------- #

def test_generic_evalprovider_matches_dedicated():
    dedicated = [c.model_dump() for c in AllganizeEvalSetProvider().load()]
    generic = [c.model_dump() for c in allganize_eval_provider().load()]
    assert len(dedicated) == len(generic) == 40
    assert dedicated == generic  # every EvalCase, every field


# --- B2: value_present ------------------------------------------------------- #

def test_generic_value_present_matches_dedicated(cases, run_records):
    generic_vp = make_value_present(allganize_gold_matcher())
    assert len(cases) == 40
    for c in cases:
        rec = run_records[c["id"]]
        assert allganize_value_present(c, rec) == generic_vp(c, rec)
