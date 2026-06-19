"""Wiki mini-domain single baseline run (G2.2-G2.3): RAG → score/attribute →
run-log + attribution + metrics. Thin wrapper over app.adapters.wiki.evaluate_wiki
(the attribution/gate_fields engine is shared with DART; only wiki hooks injected).

For the full gate demo (baseline + noise band + candidate + verdict) see
scripts/run_wiki_gate_demo.py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.wiki import (WIKI_CONFIG, build_wiki_index, evaluate_wiki,
                               wiki_metrics)

OUT = Path("examples/wiki_baseline")


def main() -> int:
    n_chunks = build_wiki_index(WIKI_CONFIG)
    print(f"wiki index: {n_chunks} chunks  embedding={WIKI_CONFIG.embedding_model}  "
          f"collection={WIKI_CONFIG.collection_name()}")

    cases, records, attrs = evaluate_wiki(WIKI_CONFIG)
    run_id = f"wiki_{datetime.now():%Y%m%d_%H%M%S}_{WIKI_CONFIG.index_signature()}"
    header = {"type": "header", "run_id": run_id, "domain": "wiki",
              "embedding_model": WIKI_CONFIG.embedding_model,
              "generation_model": WIKI_CONFIG.generation_model,
              "n_cases": len(cases), "config": WIKI_CONFIG.fingerprint()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *records]) + "\n", encoding="utf-8")
    (OUT / "attribution.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in attrs) + "\n", encoding="utf-8")
    m = wiki_metrics(attrs)
    (OUT / "metrics.json").write_text(json.dumps({"run_id": run_id, **m}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(header, ensure_ascii=False, indent=2), encoding="utf-8")

    _report(m, attrs, {r["id"]: r for r in records}, cases)
    return 0


def _report(m, attrs, rec_by, cases):
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
    for s in ("factoid", "no_answer"):
        print(f"    {s:10} {dict(by_slice[s])}")
    over = [a for a in attrs if not a["answerable"] and a["over_answer"]]
    print(f"\n  no_answer 중 over_answer: {len(over)}")
    for a in over:
        print(f"    [{a['id']}] {cases[a['id']]['question']}  -> {rec_by[a['id']]['answer'][:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
