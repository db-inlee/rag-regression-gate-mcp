"""G2.4 — engine-reuse proof: run the SAME regression gate on the wiki domain.

1. measure a wiki noise band (N runs, baseline config)  -> examples/wiki_baseline/
2. build a weakened candidate (top_k 3->1, search degraded) -> examples/wiki_candidate/
3. judge baseline vs candidate by calling the ENGINE directly:
       app.regression.detect.detect_paths  +  app.regression.gate.evaluate/render
   (importing & running the unchanged engine IS the reuse proof.)

The engine is gold-free / domain-free: it consumes only the enriched attribution
(gate_fields) + noise_band — identical format to DART. No engine edits.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.wiki import (WIKI_CONFIG, build_wiki_index, evaluate_wiki,
                               wiki_band_vector, wiki_metrics)
from app.regression.detect import detect_paths
from app.regression.gate import evaluate, exit_code, render

BASE = Path("examples/wiki_baseline")
CAND = Path("examples/wiki_candidate")
N_RUNS = 3


def _write_run(out: Path, run_id: str, config, cases, records, attrs) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    header = {"type": "header", "run_id": run_id, "domain": "wiki",
              "embedding_model": config.embedding_model, "generation_model": config.generation_model,
              "n_cases": len(cases), "config": config.fingerprint()}
    (out / "run.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *records]) + "\n", encoding="utf-8")
    (out / "attribution.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in attrs) + "\n", encoding="utf-8")
    m = wiki_metrics(attrs)
    (out / "metrics.json").write_text(json.dumps({"run_id": run_id, **m}, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def main() -> int:
    from openai import OpenAI

    from app.env import require_openai_key

    require_openai_key()
    client = OpenAI()
    build_wiki_index(WIKI_CONFIG)

    # 1) baseline noise band: N runs at baseline config (temp=0+seed → honest small band)
    print(f"[1] measuring wiki noise band: {N_RUNS} runs @ top_k={WIKI_CONFIG.top_k} ...")
    vectors, run_ids, first = [], [], None
    for r in range(1, N_RUNS + 1):
        cases, records, attrs = evaluate_wiki(WIKI_CONFIG, client=client)
        vectors.append(wiki_band_vector(attrs))
        rid = f"wiki_noise_{WIKI_CONFIG.index_signature()}_r{r}"
        run_ids.append(rid)
        if first is None:
            first = (cases, records, attrs)  # run 1 = canonical baseline

    metric_band = {k: {"mean": round(statistics.mean(v[k] for v in vectors), 4),
                       "std": round(statistics.pstdev(v[k] for v in vectors), 4),
                       "min": min(v[k] for v in vectors), "max": max(v[k] for v in vectors),
                       "n": N_RUNS} for k in vectors[0]}
    cases, records, attrs = first
    base_m = _write_run(BASE, run_ids[0], WIKI_CONFIG, cases, records, attrs)
    (BASE / "noise_band.json").write_text(json.dumps(
        {"n_runs": N_RUNS, "domain": "wiki", "config": WIKI_CONFIG.fingerprint(),
         "run_ids": run_ids, "metric_band": metric_band}, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE / "meta.json").write_text(json.dumps({"run_id": run_ids[0], "domain": "wiki",
        "config": WIKI_CONFIG.fingerprint(), "n_cases": len(cases)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    baseline retrieval_miss={base_m['retrieval_miss_count']}  "
          f"answerable_acc={base_m['answerable_accuracy']}  band(std≈0 → 1-case floor)")

    # 2) candidate: weaken retrieval (top_k 3 -> 1). Same index (top_k is query-time).
    cand_cfg = WIKI_CONFIG.model_copy(update={"top_k": 1})
    print(f"[2] candidate run @ top_k={cand_cfg.top_k} (search weakened) ...")
    c_cases, c_records, c_attrs = evaluate_wiki(cand_cfg, client=client)
    cand_m = _write_run(CAND, f"wiki_cand_topk1_{cand_cfg.index_signature()}", cand_cfg,
                        c_cases, c_records, c_attrs)
    print(f"    candidate retrieval_miss={cand_m['retrieval_miss_count']}  "
          f"answerable_acc={cand_m['answerable_accuracy']}")

    # 3) ENGINE (unchanged): detect → gate → render
    print("\n[3] running the UNCHANGED engine on the wiki domain ...")
    report = detect_paths(BASE, CAND)
    gate = evaluate(report)
    text = render(report, gate)
    print(text)
    out = Path("gate_runs") / "gate_wiki_candidate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gate": gate["status"], **report, "gate_detail": gate},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nretrieval_miss  baseline={base_m['retrieval_miss_count']}  →  candidate={cand_m['retrieval_miss_count']}  "
          f"(judge-free doc_id ⊆ detection)")
    print(f"gate: {gate['status']}   exit_code={exit_code(gate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
