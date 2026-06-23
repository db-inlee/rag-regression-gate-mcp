"""T4 — gold-free / attribution artifact contract.

The gate is CONSUMPTION-ONLY: it judges from run.jsonl + attribution.jsonl, never
from the original gold/eval-set. These tests pin that contract (Phase-6 gold removal
regression guard) and the per-case fields the engine requires.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.core.service import build_gate_result
from app.regression.detect import detect_paths
from app.regression.gate import evaluate

# Per-case fields the engine reads (features_from_enriched / gate_fields output).
REQUIRED_FIELDS = {
    "id", "slice", "primary_failure", "answerable", "correct",
    "value_present", "retrieval_strict_ok", "over_answer",
}
BOOL_FIELDS = {"answerable", "correct", "value_present", "retrieval_strict_ok", "over_answer"}

ALLG_BASE = Path("examples/allganize_baseline")
ALLG_CAND = Path("examples/allganize_candidate")


def _records(attr_path: Path) -> list[dict]:
    return [json.loads(l) for l in attr_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_detect_paths_gold_free(tmp_path):
    """detect_paths judges from attribution.jsonl (+noise_band) ALONE — no run.jsonl,
    no eval_cases/gold present. Proves the consumption-only contract."""
    base, cand = tmp_path / "base", tmp_path / "cand"
    base.mkdir(); cand.mkdir()
    # copy ONLY the enriched attribution (+ baseline noise band). No run.jsonl, no gold.
    shutil.copyfile(ALLG_BASE / "attribution.jsonl", base / "attribution.jsonl")
    shutil.copyfile(ALLG_BASE / "noise_band.json", base / "noise_band.json")
    shutil.copyfile(ALLG_CAND / "attribution.jsonl", cand / "attribution.jsonl")

    assert not (base / "run.jsonl").exists()       # no run-log
    assert not (base / "eval_cases.jsonl").exists()  # no gold
    assert not (tmp_path / "data").exists()

    report = detect_paths(base, cand)
    gate = evaluate(report)
    # same verdict as judging the real dirs → gold/run-log genuinely not needed
    assert gate["status"] == build_gate_result(str(ALLG_BASE), str(ALLG_CAND)).verdict == "FAIL"


@pytest.mark.parametrize("attr_path", [
    ALLG_BASE / "attribution.jsonl",
    Path("examples/baseline") / "attribution.jsonl",   # DART domain — contract is cross-domain
])
def test_attribution_fields_present(attr_path):
    """Every attribution record carries the gate's required per-case fields, typed right."""
    records = _records(attr_path)
    assert records, f"no records in {attr_path}"
    for r in records:
        missing = REQUIRED_FIELDS - r.keys()
        assert not missing, f"{attr_path} record {r.get('id')} missing {missing}"
        assert isinstance(r["primary_failure"], str) and r["primary_failure"]
        for f in BOOL_FIELDS:
            assert isinstance(r[f], bool), f"{r.get('id')}.{f} not bool: {r[f]!r}"
