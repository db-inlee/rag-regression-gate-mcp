"""Phase 5.1 — generate candidate runs by changing ONE config param (no code change).

Scenarios (vs baseline = noise r1, config: chunk_size 1000 / overlap 150 / top_k 5 / reranker off):
  A-1  chunk_size 1000 -> 400   (intended search regression; 200 would need ~43min
                                 reindex > 30min cap, so relaxed to 400 — reported)
  A-2  top_k 5 -> 1             (only if A-1 does NOT FAIL — stronger search regression)
  B    reranker off -> on       (fetch reranker_fetch_k=20 -> rerank -> top_k=5)
  C    chunk_overlap 150 -> 155 (neutral)

A config-signature change triggers reindexing (Phase 1.3); reranker/top_k reuse the
baseline index. The A-1 -> A-2 branch is decided in-code via the Phase 3 gate so the
run never stops mid-way.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.evaluator import metrics as M
from app.evaluator.attribution import attribute  # noqa: F401 (used via _one_run)
from app.rag.index import _collection_exists, build_index
from app.rag.pipeline import Pipeline
from app.regression.detect import detect
from app.regression.gate import evaluate
from scripts.measure_noise import _one_run  # reuse generation+score+judge+attribute path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_scenario")

A_CHUNK_SIZE = 400  # relaxed from 200 (>30min reindex); reported in summary


def _baseline_run() -> str:
    return json.loads(Path("reports/noise_band.json").read_text(encoding="utf-8"))["run_ids"][0]


def _ensure_index(config: RagConfig) -> bool:
    """Build the index if this config signature has none. Returns True if reindexed."""
    n = _collection_exists(config)
    if n > 0:
        logger.info("index reuse: sig=%s (%d chunks)", config.index_signature(), n)
        return False
    logger.info("reindexing for sig=%s ...", config.index_signature())
    build_index(config)
    return True


def run_candidate(name: str, config: RagConfig, cases: dict, client) -> dict:
    t0 = time.perf_counter()
    reindexed = _ensure_index(config)
    pipe = Pipeline(config)
    run_id = f"cand_{name}_{config.index_signature()}"
    if config.reranker_enabled:
        logger.info("[%s] reranker ON: fetch %d → rerank → top_k %d "
                    "(후보풀 확대+reranking 혼합 — baseline은 fetch=%d)",
                    name, config.reranker_fetch_k, config.top_k, DEFAULT_CONFIG.top_k)
    m, _, tot = _one_run(pipe, cases, config, client, run_id)
    elapsed = time.perf_counter() - t0
    cost = tot["prompt_tokens"] / 1e6 * M._PRICE_IN + tot["completion_tokens"] / 1e6 * M._PRICE_OUT
    logger.info("[%s] run_id=%s reindexed=%s wall=%.1fs gen_cost~$%.4f (judge gpt-4o 별도)",
                name, run_id, reindexed, elapsed, cost)
    return {"name": name, "run_id": run_id, "config": config.fingerprint(),
            "reindexed": reindexed, "metrics": m, "wall_s": round(elapsed, 1),
            "gen_cost_usd": round(cost, 4)}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    from app.env import require_openai_key
    require_openai_key()
    from openai import OpenAI
    client = OpenAI()

    cases = {c["id"]: c for c in (json.loads(l) for l in
             Path("data/eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    baseline_run = _baseline_run()
    logger.info("baseline run = %s  | A chunk_size=%d (200→400 완화: 200은 ~43분>30분)",
                baseline_run, A_CHUNK_SIZE)

    results = []

    if which in ("A", "all"):
        a1 = run_candidate("A1_chunk%d" % A_CHUNK_SIZE, RagConfig(chunk_size=A_CHUNK_SIZE), cases, client)
        results.append(a1)
        # branch via Phase 3 gate
        gate_a1 = evaluate(detect(baseline_run, a1["run_id"]))
        logger.info("[A-1 branch] gate=%s", gate_a1["status"])
        if gate_a1["status"] != "FAIL":
            logger.info("[A-1 branch] 회귀 불충분(FAIL 아님) → A-2(top_k 5→1) 진행")
            results.append(run_candidate("A2_topk1", RagConfig(top_k=1), cases, client))
        else:
            logger.info("[A-1 branch] A-1 FAIL → A-2 생략")

    if which in ("B", "all"):
        results.append(run_candidate("B_reranker", RagConfig(reranker_enabled=True), cases, client))

    if which in ("C", "all"):
        results.append(run_candidate("C_overlap155", RagConfig(chunk_overlap=155), cases, client))

    out = Path("reports/scenarios.json")
    out.write_text(json.dumps({"baseline_run": baseline_run, "a_chunk_size": A_CHUNK_SIZE,
                               "scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("===== SCENARIOS DONE =====")
    for r in results:
        m = r["metrics"]
        logger.info("  %-16s run=%s ans_acc=%s retrieval_miss=%s reindexed=%s wall=%.0fs",
                    r["name"], r["run_id"], m["answerable_accuracy"],
                    m["failure_distribution"].get("retrieval_miss"), r["reindexed"], r["wall_s"])
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
