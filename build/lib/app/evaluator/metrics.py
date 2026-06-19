"""Metric aggregation with anti-illusion guards (Phase 2.4).

Aggregates existing artifacts (run log + attribution + judge_validation + gold) —
no LLM calls. Two structural guards:

1. no_answer illusion: answerable_accuracy and no_answer_accuracy are reported as
   a PAIR, never a single composite (a refuse-everything system must show up).
2. groundedness: a "correct" answer only counts toward the headline
   answerable_accuracy if the gold value is actually present in retrieved chunks
   (grounded_correct). Lucky/memorized correct answers (unsupported_correct) are
   reported separately so model recall doesn't inflate the RAG score.

retrieval_success is reported under two definitions:
  * strict      : the gold table/page (source_ref) was retrieved (§2.3 matching)
  * value_present: the gold value appears in ANY retrieved chunk (equivalent
                   tables like 요약재무정보 also count) — case_038 style.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.evaluator.attribution import _numbers_in, gold_keys, is_retrieval_miss, retrieved_keys
from app.evaluator.scorer import parse_signed_number

EVAL_PATH = Path("data/eval_cases.jsonl")
RUNS_DIR = Path("reports/runs")
_PRICE_IN, _PRICE_OUT = 0.15, 0.60  # gpt-4o-mini (generation), USD per 1M tokens


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= 0.001 * abs(b) if b else a == b


def _retrieved_numbers(rec: dict) -> set[float]:
    nums: set[float] = set()
    for c in rec.get("retrieved_chunks", []):
        nums |= _numbers_in(c.get("text", ""))
    return nums


def _derived_source_values(case: dict) -> list[float] | None:
    """The two source cells a derived (%) answer is computed from, parsed from the
    gold snippet (contexts[0]) using the period 'prev→cur' and the year headers."""
    period = case["expected_answer"].get("period", "")
    if "→" not in period or not case.get("contexts"):
        return None
    prev_y, cur_y = (p.strip() for p in period.split("→"))
    lines = [l for l in case["contexts"][0].splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return None
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    colmap = {m.group(1): i for i, h in enumerate(header) if (m := re.search(r"\((\d{4})\)", h))}
    row = [c.strip() for c in lines[-1].strip().strip("|").split("|")]
    out: list[float] = []
    for y in (prev_y, cur_y):
        i = colmap.get(y)
        v = parse_signed_number(row[i]) if (i is not None and i < len(row)) else None
        if v is None:
            return None
        out.append(v)
    return out


def value_present(case: dict, rec: dict) -> bool:
    """Is the gold answer value findable in the retrieved chunks (any table)?"""
    schema = case["answer_schema"]
    if schema == "numeric":
        ea = case["expected_answer"]
        if ea.get("unit") == "%":  # derived: grounded if BOTH source cells retrieved
            src = _derived_source_values(case)
            if not src:
                return False
            nums = _retrieved_numbers(rec)
            return all(any(_near(n, s) for n in nums) for s in src)
        return any(_near(n, ea["value"]) for n in _retrieved_numbers(rec))
    if schema == "comparison":
        nums = _retrieved_numbers(rec)
        return all(any(_near(n, cv["value"]) for n in nums)
                   for cv in case["expected_answer"]["companies"].values())
    if schema == "text":  # proxy: the gold paragraph (page) was retrieved
        return any(g in retrieved_keys(rec) for g in gold_keys(case))
    return False  # no_answer


def aggregate(run_id: str | None = None) -> dict:
    logf = (RUNS_DIR / f"{run_id}.jsonl") if run_id else Path(sorted(glob.glob(str(RUNS_DIR / "run_*.jsonl")))[-1])
    lines = [json.loads(l) for l in logf.read_text(encoding="utf-8").splitlines() if l.strip()]
    header = next((l for l in lines if l.get("type") == "header"), {})
    records = {l["id"]: l for l in lines if l.get("type") == "case"}

    attr_path = Path("reports") / f"attribution_{logf.stem}.jsonl"
    attr = {a["id"]: a["primary_failure"]
            for a in (json.loads(l) for l in attr_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    jv_path = Path("reports/judge_validation.json")
    judge_acc = json.loads(jv_path.read_text(encoding="utf-8"))["judge_accuracy"] if jv_path.exists() else None

    cases = {c["id"]: c for c in (json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}

    ans_total = grounded = unsupported = 0
    na_total = na_correct = over = 0
    rs_strict = rs_value = 0
    by_slice = defaultdict(lambda: {"total": 0, "correct": 0, "grounded": 0})
    op = defaultdict(float)

    for cid, case in cases.items():
        rec = records.get(cid, {})
        correct = attr.get(cid) == "correct"
        answerable = case["answer_type"] == "answerable"
        sl = by_slice[case["slice"]]
        sl["total"] += 1
        sl["correct"] += correct

        # operational (generation run)
        op["latency_ms"] += rec.get("latency_ms") or 0
        op["llm_calls"] += rec.get("llm_calls") or 0
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            op[k] += (rec.get("token_usage") or {}).get(k, 0)

        if answerable:
            ans_total += 1
            grounded_here = value_present(case, rec)
            if not is_retrieval_miss(case, rec):
                rs_strict += 1
            if grounded_here:
                rs_value += 1
            if correct:
                if grounded_here:
                    grounded += 1
                    sl["grounded"] += 1
                else:
                    unsupported += 1
        else:  # no_answer
            na_total += 1
            na_correct += correct
            if attr.get(cid) == "hallucination":  # over_answer
                over += 1

    dist = Counter(attr.values())
    metrics = {
        "run_id": header.get("run_id"),
        "answerable_accuracy": round(grounded / ans_total, 4) if ans_total else None,           # headline (grounded)
        "answerable_accuracy_raw": round((grounded + unsupported) / ans_total, 4) if ans_total else None,
        "grounded_correct": grounded,
        "unsupported_correct": unsupported,                                                     # memorized/lucky
        "answerable_total": ans_total,
        "no_answer_accuracy": round(na_correct / na_total, 4) if na_total else None,
        "over_answer_rate": round(over / na_total, 4) if na_total else None,
        "retrieval_success_strict": round(rs_strict / ans_total, 4) if ans_total else None,
        "retrieval_success_value_present": round(rs_value / ans_total, 4) if ans_total else None,
        "per_slice": {s: {**v, "accuracy": round(v["correct"] / v["total"], 4)} for s, v in by_slice.items()},
        "failure_distribution": dict(dist),
        "judge_accuracy": judge_acc,
        "operational": {
            "avg_latency_ms": round(op["latency_ms"] / max(len(records), 1), 1),
            "total_llm_calls": int(op["llm_calls"]),
            "total_tokens": int(op["total_tokens"]),
            "est_cost_usd": round(op["prompt_tokens"] / 1e6 * _PRICE_IN + op["completion_tokens"] / 1e6 * _PRICE_OUT, 4),
        },
    }
    return metrics


def _print(m: dict) -> None:
    print("\n===== PHASE 2 METRICS =====  run_id:", m["run_id"])
    print("── answerable vs no_answer (짝, 착시 방어) ──")
    print(f"  answerable_accuracy (grounded)   : {m['answerable_accuracy']}  "
          f"({m['grounded_correct']}/{m['answerable_total']})")
    print(f"    └ raw correct (grounded+암기)   : {m['answerable_accuracy_raw']}  "
          f"(unsupported_correct={m['unsupported_correct']})")
    print(f"  no_answer_accuracy               : {m['no_answer_accuracy']}  (over_answer_rate={m['over_answer_rate']})")
    print("── retrieval ──")
    print(f"  retrieval_success_strict         : {m['retrieval_success_strict']}  (gold table/page)")
    print(f"  retrieval_success_value_present  : {m['retrieval_success_value_present']}  (정답값이 어느 청크에든)")
    print("── per-slice (total/correct/grounded · acc) ──")
    for s, v in m["per_slice"].items():
        print(f"  {s:18} {v['total']:>3}/{v['correct']:>3}/{v['grounded']:>3}   acc={v['accuracy']}")
    print("── failure modes ──")
    for k, n in m["failure_distribution"].items():
        print(f"  {k:18} {n}")
    print(f"── judge_accuracy: {m['judge_accuracy']}")
    o = m["operational"]
    print(f"── ops: avg_latency={o['avg_latency_ms']}ms  llm_calls={o['total_llm_calls']}  "
          f"tokens={o['total_tokens']}  ~${o['est_cost_usd']}")


def main() -> int:
    m = aggregate()
    out = Path("reports") / f"metrics_{m['run_id']}.json"
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(m)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
