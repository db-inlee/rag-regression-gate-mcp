"""Wiki mini-domain baseline (G2.2-G2.3): run the 20 cases through the wiki RAG,
attribute with the wiki plugins injected, and emit run-log + attribution + metrics.

Demonstrates the SAME attribution/gate-field machinery driving a second domain via
plugin injection only (WikiGoldMatcher + wiki_value_present). The engine
(detect/gate/metrics/noise_band) is untouched — metrics here are derived from the
portable gate_fields (the same booleans the gate consumes).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.wiki import (WIKI_CONFIG, WikiEvalSetProvider, WikiGoldMatcher,
                               WikiRAGAdapter, WikiScoringPlugin, build_wiki_index,
                               wiki_value_present)
from app.evaluator.attribution import attribute
from app.evaluator.case_eval import gate_fields

OUT = Path("examples/wiki_baseline")


def _metrics_from_gate_fields(gfs: list[dict]) -> dict:
    """Same metric DEFINITIONS as the DART engine, computed from the portable
    gate_fields (answerable/correct/value_present/retrieval_strict_ok/over_answer)."""
    ans = [g for g in gfs if g["answerable"]]
    na = [g for g in gfs if not g["answerable"]]
    grounded = [g for g in ans if g["correct"] and g["value_present"]]
    return {
        "answerable_accuracy": round(len(grounded) / len(ans), 4) if ans else None,  # grounded
        "answerable_total": len(ans),
        "grounded_correct": len(grounded),
        "unsupported_correct": sum(1 for g in ans if g["correct"] and not g["value_present"]),
        "no_answer_accuracy": round(sum(g["correct"] for g in na) / len(na), 4) if na else None,
        "over_answer_rate": round(sum(g["over_answer"] for g in na) / len(na), 4) if na else None,
        "retrieval_success_strict": round(sum(g["retrieval_strict_ok"] for g in ans) / len(ans), 4) if ans else None,
        "retrieval_miss_count": sum(1 for g in gfs if g["primary_failure"] == "retrieval_miss"),
        "failure_distribution": dict(Counter(g["primary_failure"] for g in gfs)),
    }


def main() -> int:
    from openai import OpenAI

    from app.env import require_openai_key

    cases = {c.model_dump()["id"]: c.model_dump() for c in WikiEvalSetProvider().load()}
    n_chunks = build_wiki_index(WIKI_CONFIG)
    print(f"wiki index: {n_chunks} chunks  embedding={WIKI_CONFIG.embedding_model}  "
          f"collection={WIKI_CONFIG.collection_name()}")

    adapter = WikiRAGAdapter(WIKI_CONFIG)
    scorer = WikiScoringPlugin()
    matcher = WikiGoldMatcher()
    require_openai_key()
    client = OpenAI()

    OUT.mkdir(parents=True, exist_ok=True)
    run_id = f"wiki_{datetime.now():%Y%m%d_%H%M%S}_{WIKI_CONFIG.index_signature()}"
    records, attrs = [], []
    for cid, case in cases.items():
        entry = adapter.run(case["question"])
        rec = {"type": "case", "id": cid, "slice": case["slice"], **entry}
        records.append(rec)

        scored = scorer.score(entry["answer"], case["expected_answer"], case)
        correct = scored["correct"]
        if correct is None:  # ambiguous factoid → judge (reuse DART judge on key_points)
            from app.evaluator.judge import judge_body_text
            v = judge_body_text(case["question"], case["expected_answer"]["key_points"],
                                entry["answer"], WIKI_CONFIG, client)
            correct = v["correct"]
            scored["score_detail"]["judge_reason"] = v["reason"]

        a = attribute(case, correct, scored, entry, WIKI_CONFIG, client=client, matcher=matcher)
        a.update(gate_fields(case, entry, a["primary_failure"],
                             matcher=matcher, value_present=wiki_value_present))
        attrs.append(a)

    header = {"type": "header", "run_id": run_id, "domain": "wiki",
              "embedding_model": WIKI_CONFIG.embedding_model,
              "generation_model": WIKI_CONFIG.generation_model,
              "n_cases": len(cases), "config": WIKI_CONFIG.fingerprint()}
    (OUT / "run.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *records]) + "\n", encoding="utf-8")
    (OUT / "attribution.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in attrs) + "\n", encoding="utf-8")
    metrics = _metrics_from_gate_fields(attrs)
    (OUT / "metrics.json").write_text(json.dumps({"run_id": run_id, **metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(header, ensure_ascii=False, indent=2), encoding="utf-8")

    _report(metrics, attrs, records, cases)
    return 0


def _report(m, attrs, records, cases):
    rec_by = {r["id"]: r for r in records}
    by_slice = defaultdict(Counter)
    for a in attrs:
        by_slice[a["slice"]][a["primary_failure"]] += 1

    print("\n===== WIKI BASELINE METRICS (same definitions as DART) =====")
    print(f"  answerable_accuracy (grounded)  : {m['answerable_accuracy']}  "
          f"({m['grounded_correct']}/{m['answerable_total']})  unsupported={m['unsupported_correct']}")
    print(f"  no_answer_accuracy              : {m['no_answer_accuracy']}  (over_answer_rate={m['over_answer_rate']})")
    print(f"  retrieval_success_strict       : {m['retrieval_success_strict']}")
    print(f"  retrieval_miss_count           : {m['retrieval_miss_count']}")
    print(f"  failure_distribution           : {m['failure_distribution']}")
    print("  by slice:")
    for s in ("factoid", "no_answer"):
        print(f"    {s:10} {dict(by_slice[s])}")

    over = [a for a in attrs if not a["answerable"] and a["over_answer"]]
    print(f"\n  no_answer 케이스 중 over_answer(모델이 함부로 답함): {len(over)}")
    for a in over:
        print(f"    [{a['id']}] Q: {cases[a['id']]['question']}")
        print(f"        answer: {rec_by[a['id']]['answer'][:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
