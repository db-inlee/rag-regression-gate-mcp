"""Phase A verification — refactor must not change DART behavior.

Recomputes DART attribution over the EXISTING baseline run-log via the REFACTORED
code (default = DART matcher/value_present), reusing judge verdicts for body_text
(judge isn't deterministic / not wrapped). Compares the gate fields to
examples/baseline/attribution.jsonl. Deterministic — no LLM.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.dart import DartEvalSetProvider
from app.config import DEFAULT_CONFIG
from app.evaluator.attribution import attribute
from app.evaluator.case_eval import gate_fields
from app.evaluator.scorer import score_case

BASE = Path("examples/baseline")
OUT = Path("gate_runs/phaseA_check")
GATE_FIELDS = ["id", "slice", "answerable", "correct", "value_present",
               "retrieval_strict_ok", "over_answer", "primary_failure"]


def main() -> int:
    cases = {c.model_dump()["id"]: c.model_dump() for c in DartEvalSetProvider().load()}  # +whitelist
    recs = {r["id"]: r for r in (json.loads(l) for l in (BASE / "run.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if r.get("type") == "case"}
    base_attr = {a["id"]: a for a in (json.loads(l) for l in (BASE / "attribution.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    new = []
    for cid, case in cases.items():
        rec = recs[cid]
        if case["answer_schema"] == "text":          # reuse judge verdict (not deterministic)
            correct, scored = base_attr[cid]["correct"], {"score_detail": {}}
        else:                                         # deterministic re-score (refactored)
            scored = score_case(case, rec["answer"])
            correct = scored["correct"]
        a = attribute(case, correct, scored, rec, DEFAULT_CONFIG)      # refactored, default DART
        a.update(gate_fields(case, rec, a["primary_failure"]))         # refactored, default DART
        new.append(a)

    new_by = {a["id"]: a for a in new}
    diff = [cid for cid in cases if {k: new_by[cid].get(k) for k in GATE_FIELDS} != {k: base_attr[cid].get(k) for k in GATE_FIELDS}]
    cmp_ids = [cid for cid, c in cases.items() if c["answer_schema"] == "comparison"]
    cmp_diff = [cid for cid in cmp_ids if cid in diff]

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(BASE / "run.jsonl", OUT / "run.jsonl")
    shutil.copy(BASE / "noise_band.json", OUT / "noise_band.json") if (BASE / "noise_band.json").exists() else None
    (OUT / "attribution.jsonl").write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in new) + "\n", encoding="utf-8")

    print(f"cases: {len(cases)}  (DartEvalSetProvider.load() + whitelist OK)")
    print(f"attribution gate-field diff vs examples/baseline: {len(diff)}  {diff[:5]}")
    print(f"comparison(8) gate-field diff: {len(cmp_diff)} / {len(cmp_ids)}  {cmp_diff}")
    print(f"wrote {OUT}/ → run_gate.py --baseline examples/baseline --candidate {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
