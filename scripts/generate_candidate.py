"""Phase 6.2 — Job 1 (RAG run) entrypoint: produce a candidate the gate can read.

Heavy path (LLM + embedding) — runs OUTSIDE the PR gate. Given the current config,
it: ensures the signature index (bge-m3) → runs the pipeline over the 100 eval
cases → scores (deterministic) + judges body_text (gpt-4o) + attributes → writes
the SELF-CONTAINED candidate artifacts (run.jsonl + ENRICHED attribution.jsonl +
meta.json) to --out. Output goes to gate_runs/ (a byproduct dir), NOT reports/.

Then `run_gate.py --baseline examples/baseline --candidate <out>` gates it.

Requires OPENAI_API_KEY (judge). If unset, it prints a notice and skips (exit 0) —
so the skeleton workflow stays green without a key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_CONFIG
from app.env import load_dotenv
from app.evaluator.attribution import attribute
from app.evaluator.case_eval import gate_fields
from app.evaluator.judge import judge_body_text
from app.evaluator.scorer import score_case


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gate_runs/candidate", help="candidate output dir")
    args = ap.parse_args()
    config = DEFAULT_CONFIG

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 미설정 → candidate 생성 스킵(안내). "
              "Model 2(완전 자동)는 secrets.OPENAI_API_KEY가 필요합니다.")
        return 0

    from openai import OpenAI

    from app.rag.index import _collection_exists, build_index
    from app.rag.pipeline import Pipeline

    cases = {c["id"]: c for c in (json.loads(l) for l in
             Path("data/eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    if _collection_exists(config) <= 0:           # 1) index (bge-m3, signature collection)
        print(f"indexing for sig={config.index_signature()} ...")
        build_index(config)

    pipe = Pipeline(config)                        # 2) retrieve+generate
    client = OpenAI()
    runs, attrs = [], []
    for cid, case in cases.items():
        r = pipe.run(case["question"])
        rec = {"type": "case", "id": cid, "slice": case["slice"], "question": case["question"],
               "retrieved_chunks": r["retrieved_chunks"], "answer": r["answer"],
               "latency_ms": r["latency_ms"], "llm_calls": r["llm_calls"], "token_usage": r["token_usage"]}
        runs.append(rec)
        if case["answer_schema"] == "text":        # 3) score + judge body_text
            v = judge_body_text(case["question"], case["expected_answer"]["key_points"], r["answer"], config, client)
            correct, scored = v["correct"], {"score_detail": {"judge_reason": v["reason"]}}
        else:
            scored = score_case(case, r["answer"])
            correct = scored["correct"]
        a = attribute(case, correct, scored, rec, config, client)   # 4) attribute (+ gate fields)
        a.update(gate_fields(case, rec, a["primary_failure"]))
        attrs.append(a)

    out = Path(args.out)                            # 5) package self-contained candidate
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in runs) + "\n", encoding="utf-8")
    (out / "attribution.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in attrs) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(
        {"role": "generated-candidate", "config": config.fingerprint(),
         "generated": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidate 생성 완료 → {out}/  (run.jsonl + attribution.jsonl). "
          f"게이트: python scripts/run_gate.py --baseline examples/baseline --candidate {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
