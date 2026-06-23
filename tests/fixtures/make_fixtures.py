"""Generate the synthetic gate fixtures used by tests/test_gate_behavior.py.

Deterministic, reproducible. Run from the repo root:
    python tests/fixtures/make_fixtures.py

Fixtures (derived from examples/allganize_baseline — NOT new RAG runs):

  allganize_candidate_1case/
      A copy of the baseline run-log/attribution with exactly ONE answerable case
      flipped correct → hallucination (answerable_accuracy −1/40 = −0.025). This is a
      change INSIDE the noise band (baseline answerable_accuracy std ≈ 0.02) and at
      the ±1-case floor, so the gate must NOT FAIL on it (false-alarm guard).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

BASE = Path("examples/allganize_baseline")
FIX = Path("tests/fixtures")


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def make_one_case_candidate() -> Path:
    """Baseline with ONE correct answerable case flipped to hallucination."""
    out = FIX / "allganize_candidate_1case"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE / "run.jsonl", out / "run.jsonl")  # detect reads attribution; keep run for contract

    attrs = _load(BASE / "attribution.jsonl")
    flipped = False
    for a in attrs:
        if not flipped and a["answerable"] and a["correct"]:
            a["correct"] = False
            a["primary_failure"] = "hallucination"  # gold retrieved but answer wrong
            a["attribution_detail"] = {"reason": "synthetic 1-case flip (fixture)"}
            flipped = True
    assert flipped, "no correct answerable case to flip"
    _write(out / "attribution.jsonl", attrs)
    return out


if __name__ == "__main__":
    p = make_one_case_candidate()
    print(f"wrote {p} (1 answerable case flipped correct→hallucination)")
