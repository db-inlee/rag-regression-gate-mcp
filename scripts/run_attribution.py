"""Attribute a baseline run (Phase 2.3 demo/runner).

For each case: final correctness (deterministic scorer for numeric/comparison/
no_answer; LLM judge for body_text), then a single primary_failure. Prints the
failure-mode distribution + per-slice breakdown and saves per-case attribution.
"""

from __future__ import annotations

import glob
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_CONFIG
from app.evaluator.attribution import attribute, gold_keys, retrieved_keys
from app.evaluator.case_eval import gate_fields
from app.evaluator.judge import judge_body_text
from app.evaluator.scorer import score_case

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_attribution")


def main() -> int:
    config = DEFAULT_CONFIG
    cases = {c["id"]: c for c in (json.loads(l) for l in
             Path("data/eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    logf = sorted(glob.glob("reports/runs/run_*.jsonl"))[-1]
    records = {r["id"]: r for r in (json.loads(l) for l in
               Path(logf).read_text(encoding="utf-8").splitlines() if l.strip())
               if r.get("type") == "case"}
    logger.info("attributing %s (%d cases)", logf, len(records))

    from openai import OpenAI
    from app.env import require_openai_key
    require_openai_key()
    client = OpenAI()

    attributions = []
    for cid, case in cases.items():
        rec = records.get(cid, {})
        ans = rec.get("answer")
        if case["answer_schema"] == "text":  # body_text -> judge
            verdict = judge_body_text(case["question"], case["expected_answer"]["key_points"], ans, config, client)
            correct = verdict["correct"]
            scored = {"score_detail": {"judge_reason": verdict["reason"]}}
        else:
            scored = score_case(case, ans)
            correct = scored["correct"]
        a = attribute(case, correct, scored, rec, config, client)
        a.update(gate_fields(case, rec, a["primary_failure"]))  # self-contained gate input
        attributions.append(a)

    _report(attributions, cases, records, logf)
    return 0


def _report(attrs, cases, records, logf):
    dist = Counter(a["primary_failure"] for a in attrs)
    by_slice = defaultdict(Counter)
    for a in attrs:
        by_slice[a["slice"]][a["primary_failure"]] += 1

    logger.info("===== FAILURE ATTRIBUTION =====")
    logger.info("primary_failure distribution (n=%d):", len(attrs))
    for mode in ("correct", "retrieval_miss", "table_value_error", "hallucination"):
        logger.info("  %-18s %d", mode, dist.get(mode, 0))
    logger.info("by slice:")
    for s in ("table_value", "numeric_reasoning", "body_text", "no_answer"):
        logger.info("  %-18s %s", s, dict(by_slice[s]))

    logger.info("--- 3 retrieval_miss 예시 (gold table_id vs retrieved) ---")
    rmiss = [a for a in attrs if a["primary_failure"] == "retrieval_miss"][:3]
    for a in rmiss:
        gks = [f"{f}:{t}" for kind, f, t in gold_keys(cases[a["id"]]) if kind == "table"] \
              or [f"{f}:p{p}" for kind, f, p in gold_keys(cases[a["id"]]) if kind == "page"]
        ret = sorted({f"{m['metadata'].get('source_file')}:{m['metadata'].get('table_id')}"
                      for m in records[a["id"]].get("retrieved_chunks", []) if m["metadata"].get("is_table")})
        logger.info("  [%s %s] gold=%s", a["id"], a["slice"], gks)
        logger.info("        retrieved tables=%s", ret or "(none table)")

    out = Path(logf).with_suffix("").name
    p = Path("reports") / f"attribution_{out}.jsonl"
    p.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in attrs) + "\n", encoding="utf-8")
    logger.info("wrote %s", p)


if __name__ == "__main__":
    raise SystemExit(main())
