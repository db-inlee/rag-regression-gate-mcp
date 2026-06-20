"""B4 — integrated Generic demo: reconstruct Allganize's evaluation using the
CONFIG-DRIVEN Generic components and prove it equals the dedicated adapter.

  matcher       = generic.allganize_gold_matcher()        # B1 (2 lambdas)
  value_present = generic.make_value_present(matcher)      # B2 (matcher reuse)
  provider      = generic.allganize_eval_provider()        # B3 (ColumnMap)

Deterministic & LLM-free by construction: the Generic refactor only swaps the
GoldMatcher / value_present / EvalSetProvider — NOT generation or judging. So we
reuse the recorded baseline run-log (same RAG answers) and the same per-case judge
verdict, recompute attribution with the Generic plugins, and diff against the
stored dedicated attribution. Then run_gate(dedicated baseline vs Generic
candidate) → PASS lets the ENGINE itself certify diff 0 (same verification style).

Usage: python scripts/demo_generic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.allganize import ALLGANIZE_CONFIG, allganize_metrics
from app.adapters.generic import (allganize_eval_provider, allganize_gold_matcher,
                                  make_value_present)
from app.evaluator.attribution import attribute
from app.evaluator.case_eval import gate_fields
from app.regression.detect import detect_paths
from app.regression.gate import evaluate, exit_code, render

BASE = Path("examples/allganize_baseline")
CAND = Path("examples/allganize_generic")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    # 1) GENERIC components (config-driven)
    cases = {c.model_dump()["id"]: c.model_dump() for c in allganize_eval_provider().load()}
    matcher = allganize_gold_matcher()
    value_present = make_value_present(matcher)

    # 2) reuse recorded baseline run-log (RAG answers) + judge verdict per case
    recs = {o["id"]: o for o in _load_jsonl(BASE / "run.jsonl") if o.get("type") == "case"}
    stored = {a["id"]: a for a in _load_jsonl(BASE / "attribution.jsonl")}
    header = next(o for o in _load_jsonl(BASE / "run.jsonl") if o.get("type") == "header")

    # 3) recompute attribution with the GENERIC plugins (no RAG, no judge re-run:
    #    correct = the already-decided verdict; Generic changes only matcher/value_present)
    new_attrs = []
    for cid, case in cases.items():
        rec = recs[cid]
        correct = stored[cid]["correct"]
        scored = {"id": cid, "slice": case["slice"], "correct": correct, "score_detail": {}}
        a = attribute(case, correct, scored, rec, ALLGANIZE_CONFIG, client=None, matcher=matcher)
        a.update(gate_fields(case, rec, a["primary_failure"],
                             matcher=matcher, value_present=value_present))
        new_attrs.append(a)

    # 4) diff vs dedicated attribution (field-by-field, all 40)
    mism = [cid for cid in cases if new_attrs_by(new_attrs)[cid] != stored[cid]]
    print(f"attribution diff (Generic vs dedicated): {len(cases) - len(mism)}/{len(cases)} identical"
          f"{'  OK' if not mism else '  MISMATCH ' + str(mism)}")

    # 5) metrics diff
    m_new = allganize_metrics(new_attrs)
    m_old = allganize_metrics(list(stored.values()))
    print(f"metrics identical: {m_new == m_old}")
    if m_new != m_old:
        print(f"  new={m_new}\n  old={m_old}")

    # 6) write Generic candidate (reuse run-log; Generic-recomputed attribution)
    CAND.mkdir(parents=True, exist_ok=True)
    cand_header = {**header, "run_id": header["run_id"] + "_generic", "note": "generic-reconstructed"}
    (CAND / "run.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [cand_header, *recs.values()]) + "\n",
        encoding="utf-8")
    (CAND / "attribution.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in new_attrs) + "\n", encoding="utf-8")
    (CAND / "metrics.json").write_text(json.dumps({"run_id": cand_header["run_id"], **m_new},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) ENGINE certifies the refactor: dedicated baseline vs Generic candidate → PASS
    print("\n[run_gate] dedicated baseline  vs  Generic candidate (engine = judge of diff 0):")
    report = detect_paths(BASE, CAND)
    gate = evaluate(report)
    print(render(report, gate))
    out = Path("gate_runs") / "gate_allganize_generic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gate": gate["status"], **report, "gate_detail": gate},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngate: {gate['status']}   exit_code={exit_code(gate)}   "
          f"(PASS = Generic reproduces dedicated, diff 0)")
    return 0


def new_attrs_by(new_attrs: list[dict]) -> dict:
    return {a["id"]: a for a in new_attrs}


if __name__ == "__main__":
    raise SystemExit(main())
