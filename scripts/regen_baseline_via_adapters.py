"""§6 full verification — regenerate baseline through the DART wrappers and diff.

Routes the WHOLE pipeline through app.adapters.dart (EvalSetProvider + RAGAdapter +
ScoringPlugin + GoldMatcher); body_text correctness via the existing judge (not a
wrapper yet); attribution via the engine. Writes gate_runs/wrapper_baseline/ and
diffs it against examples/baseline (the original, non-wrapped run).

Proves "wrapped, not rewritten": the assembled output must match. run-log compared
semantically (retrieved keys + answer; latency is wall-time, ignored); attribution
compared on the gate fields, with comparison cases reported separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.dart import (
    DartEvalSetProvider, DartGoldMatcher, DartRAGAdapter, DartScoringPlugin,
)
from app.config import DEFAULT_CONFIG
from app.evaluator.attribution import attribute
from app.evaluator.case_eval import gate_fields
from app.evaluator.judge import judge_body_text

OUT = Path("gate_runs/wrapper_baseline")
BASE = Path("examples/baseline")
GATE_FIELDS = ["id", "slice", "answerable", "correct", "value_present",
               "retrieval_strict_ok", "over_answer", "primary_failure"]


def _chunk_keys(chunks):
    return [(c["metadata"].get("source_file"), c["metadata"].get("table_id"),
             c["metadata"].get("page")) for c in chunks]


def main() -> int:
    from app.env import require_openai_key
    require_openai_key()
    from openai import OpenAI
    client = OpenAI()

    config = DEFAULT_CONFIG
    ep, rag, sp = DartEvalSetProvider(), DartRAGAdapter(config), DartScoringPlugin()
    _ = DartGoldMatcher()  # identity already proven; engine attribution uses same funcs

    cases = ep.load()
    runs, attrs = [], []
    for case in cases:
        cd = case.model_dump()
        entry = rag.run(cd["question"])                       # DartRAGAdapter
        rec = {"type": "case", "id": cd["id"], "slice": cd["slice"], "question": cd["question"],
               "retrieved_chunks": entry["retrieved_chunks"], "answer": entry["answer"],
               "latency_ms": entry["latency_ms"], "llm_calls": entry["llm_calls"],
               "token_usage": entry["token_usage"]}
        runs.append(rec)
        if cd["answer_schema"] == "text":                    # body_text → judge (not wrapped)
            v = judge_body_text(cd["question"], cd["expected_answer"]["key_points"], entry["answer"], config, client)
            correct, scored = v["correct"], {"score_detail": {}}
        else:                                                # DartScoringPlugin
            scored = sp.score(entry["answer"], cd.get("expected_answer"), cd)
            correct = scored["correct"]
        a = attribute(cd, correct, scored, rec, config, client)
        a.update(gate_fields(cd, rec, a["primary_failure"]))
        attrs.append(a)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in runs) + "\n", encoding="utf-8")
    (OUT / "attribution.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in attrs) + "\n", encoding="utf-8")

    # --- diff vs examples/baseline ---
    base_runs = {r["id"]: r for r in (json.loads(l) for l in (BASE / "run.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if r.get("type") == "case"}
    base_attr = {a["id"]: a for a in (json.loads(l) for l in (BASE / "attribution.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    schema = {c.model_dump()["id"]: c.model_dump()["answer_schema"] for c in cases}

    run_diff = [r["id"] for r in runs
                if _chunk_keys(r["retrieved_chunks"]) != _chunk_keys(base_runs[r["id"]]["retrieved_chunks"])
                or r["answer"] != base_runs[r["id"]]["answer"]]
    attr_diff = [a["id"] for a in attrs
                 if {k: a.get(k) for k in GATE_FIELDS} != {k: base_attr[a["id"]].get(k) for k in GATE_FIELDS}]
    cmp_ids = [cid for cid, s in schema.items() if s == "comparison"]
    cmp_diff = [cid for cid in cmp_ids if cid in attr_diff]

    print("===== §6 FULL VERIFICATION (wrapper-regen vs examples/baseline) =====")
    print(f"cases: {len(runs)}")
    print(f"run-log diff (retrieved keys + answer): {len(run_diff)}  {run_diff[:5]}")
    print(f"attribution gate-field diff           : {len(attr_diff)}  {attr_diff[:5]}")
    print(f"comparison(8) attribution diff        : {len(cmp_diff)} / {len(cmp_ids)}  {cmp_diff}")
    print(f"wrote {OUT}/  → 다음: run_gate.py --baseline examples/baseline --candidate {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
