"""Phase 1.3.3 — run the 100-case eval set through the RAG pipeline and log it.

Writes reports/runs/{run_id}.jsonl:
  * line 1: header {type:"header", run_id, seed, timestamp, config fingerprint,
    generation_model, embedding_model, n_cases}  -- for reproducibility
  * one line per case: {type:"case", id, slice, question, retrieved_chunks, answer,
    latency_ms, llm_calls, token_usage, error?}

A failing case is recorded with its error and the run continues. Prints a cost/time
summary at the end (per-slice completion, total llm_calls / tokens / elapsed).
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.rag.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_eval")

EVAL_PATH = Path("data/eval_cases.jsonl")
RUNS_DIR = Path("reports/runs")

# gpt-4o-mini pricing (USD per 1M tokens) for a rough cost estimate only.
_PRICE_IN, _PRICE_OUT = 0.15, 0.60


def _run_id(config: RagConfig) -> str:
    return f"run_{datetime.now():%Y%m%d_%H%M%S}_{config.index_signature()}"


def main(config: RagConfig = DEFAULT_CONFIG) -> int:
    random.seed(config.seed)
    cases = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not cases:
        logger.error("no cases in %s", EVAL_PATH)
        return 1

    pipe = Pipeline(config)
    run_id = _run_id(config)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{run_id}.jsonl"

    header = {
        "type": "header",
        "run_id": run_id,
        "seed": config.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generation_model": config.generation_model,
        "embedding_model": config.embedding_model,
        "n_cases": len(cases),
        "config": config.fingerprint(),
    }
    logger.info("run_id=%s  cases=%d  -> %s", run_id, len(cases), out_path)
    logger.info("config: %s", json.dumps(config.fingerprint(), ensure_ascii=False))

    completed: Counter = Counter()
    errored: Counter = Counter()
    tot = defaultdict(int)  # llm_calls, prompt_tokens, completion_tokens, total_tokens
    wall0 = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i, case in enumerate(cases, 1):
            rec = {"type": "case", "id": case["id"], "slice": case["slice"],
                   "question": case["question"]}
            try:
                r = pipe.run(case["question"])
                rec.update({
                    "retrieved_chunks": r["retrieved_chunks"],
                    "answer": r["answer"],
                    "latency_ms": r["latency_ms"],
                    "llm_calls": r["llm_calls"],
                    "token_usage": r["token_usage"],
                })
                completed[case["slice"]] += 1
                tot["llm_calls"] += r["llm_calls"]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    tot[k] += r["token_usage"].get(k, 0)
            except Exception as exc:  # noqa: BLE001 - isolate per-case failure
                rec.update({"retrieved_chunks": [], "answer": None, "latency_ms": None,
                            "llm_calls": 0, "token_usage": {}, "error": repr(exc)})
                errored[case["slice"]] += 1
                logger.warning("case %s FAILED: %r", case["id"], exc)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                logger.info("  ... %d/%d", i, len(cases))

    elapsed = time.perf_counter() - wall0
    _summary(cases, completed, errored, tot, elapsed, out_path)
    return 0


def _summary(cases, completed, errored, tot, elapsed, out_path) -> None:
    total = len(cases)
    n_ok = sum(completed.values())
    cost = tot["prompt_tokens"] / 1e6 * _PRICE_IN + tot["completion_tokens"] / 1e6 * _PRICE_OUT

    logger.info("===== RUN SUMMARY =====")
    logger.info("output: %s", out_path)
    logger.info("completed %d/%d  (errored %d)", n_ok, total, sum(errored.values()))
    logger.info("by slice (완료/에러):")
    for s in ("table_value", "numeric_reasoning", "body_text", "no_answer"):
        logger.info("  %-18s %d / %d", s, completed.get(s, 0), errored.get(s, 0))
    logger.info("llm_calls total : %d", tot["llm_calls"])
    logger.info("tokens          : prompt %d + completion %d = %d",
                tot["prompt_tokens"], tot["completion_tokens"], tot["total_tokens"])
    logger.info("est. cost (USD) : ~$%.4f  (gpt-4o-mini rate, approx)", cost)
    logger.info("wall time       : %.1fs  (avg %.2fs/case)", elapsed, elapsed / max(total, 1))


if __name__ == "__main__":
    raise SystemExit(main())
