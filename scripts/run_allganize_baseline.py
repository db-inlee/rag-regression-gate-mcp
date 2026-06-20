"""Allganize (law/public) single baseline run (D2): build index → RAG → score/
attribute → run-log + attribution + metrics. Thin wrapper over
app.adapters.allganize.evaluate_allganize (attribution/gate/judge engine shared
with DART; only Allganize GoldMatcher + value-presence hook injected).

Usage: python scripts/run_allganize_baseline.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.allganize import (ALLGANIZE_CONFIG, allganize_metrics,
                                    build_allganize_index, evaluate_allganize)

OUT = Path("examples/allganize_baseline")


def main() -> int:
    n_chunks = build_allganize_index(ALLGANIZE_CONFIG)
    print(f"allganize index: {n_chunks} chunks  embedding={ALLGANIZE_CONFIG.embedding_model}  "
          f"collection={ALLGANIZE_CONFIG.collection_name()}")

    cases, records, attrs = evaluate_allganize(ALLGANIZE_CONFIG)
    run_id = f"allganize_{datetime.now():%Y%m%d_%H%M%S}_{ALLGANIZE_CONFIG.index_signature()}"
    header = {"type": "header", "run_id": run_id, "domain": "allganize",
              "embedding_model": ALLGANIZE_CONFIG.embedding_model,
              "generation_model": ALLGANIZE_CONFIG.generation_model,
              "n_cases": len(cases), "config": ALLGANIZE_CONFIG.fingerprint()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *records]) + "\n", encoding="utf-8")
    (OUT / "attribution.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in attrs) + "\n", encoding="utf-8")
    m = allganize_metrics(attrs)
    (OUT / "metrics.json").write_text(json.dumps({"run_id": run_id, **m}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(header, ensure_ascii=False, indent=2), encoding="utf-8")

    _report(m, attrs)
    return 0


def _report(m, attrs):
    by_slice = defaultdict(Counter)
    by_domain = defaultdict(Counter)
    for a in attrs:
        by_slice[a["slice"]][a["primary_failure"]] += 1
        # domain is encoded in id order; recover via slice only here, domain via attrs absent → skip
    print("\n===== ALLGANIZE BASELINE METRICS (same definitions as DART) =====")
    print(f"  answerable_accuracy (grounded)  : {m['answerable_accuracy']}  "
          f"({m['grounded_correct']}/{m['answerable_total']})  unsupported={m['unsupported_correct']}")
    print(f"  retrieval_success_strict (doc)  : {m['retrieval_success_strict']}")
    print(f"  retrieval_miss_count            : {m['retrieval_miss_count']}")
    print(f"  failure_distribution            : {m['failure_distribution']}")
    print("  by slice (primary_failure):")
    for s in ("paragraph", "table"):
        print(f"    {s:10} {dict(by_slice[s])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
