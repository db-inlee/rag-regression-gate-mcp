"""Phase 3.1 — measure the noise band by running the SAME config N times.

temperature=0 + fixed seed minimize noise but cannot remove it (LLM generation and
the gpt-4o judge are both nondeterministic). We run the full pipeline (generation +
fresh scoring/judging/attribution) N=5 times and record:
  * per-metric band: mean / std / min / max over the 5 runs
  * per-case stability: how many of the 5 runs judged each case correct
  * unstable cases: those whose verdict flipped (0 < correct_count < 5) — the
    composition wobble that aggregate bands hide (used by 3.2 paired bootstrap)

The judge is re-run on every run (not cached): judge nondeterminism is part of the
band, so we never reuse one run's verdicts for another.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.evaluator import metrics as M
from app.evaluator.attribution import attribute
from app.evaluator.case_eval import gate_fields
from app.evaluator.judge import judge_body_text
from app.evaluator.scorer import score_case
from app.rag.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("measure_noise")

N_RUNS = 5
RUNS_DIR = Path("reports/runs")
OUT_PATH = Path("reports/noise_band.json")

TRACKED = ["answerable_accuracy", "grounded_correct", "unsupported_correct",
           "retrieval_success_strict", "retrieval_success_value_present",
           "no_answer_accuracy", "over_answer_rate"]
TRACKED_MODES = ["correct", "retrieval_miss", "hallucination"]


def _one_run(pipe: Pipeline, cases: dict, config: RagConfig, client, run_id: str) -> tuple[dict, dict, dict]:
    """Generate + score + (fresh) judge + attribute one run. Returns (metrics, per_case_correct, totals)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"{run_id}.jsonl"
    header = {"type": "header", "run_id": run_id, "seed": config.seed,
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "generation_model": config.generation_model, "judge_model": config.judge_model,
              "embedding_model": config.embedding_model, "n_cases": len(cases),
              "config": config.fingerprint()}
    records: dict[str, dict] = {}
    totals: dict[str, float] = defaultdict(float)

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for case in cases.values():
            r = pipe.run(case["question"])
            rec = {"type": "case", "id": case["id"], "slice": case["slice"],
                   "question": case["question"], "retrieved_chunks": r["retrieved_chunks"],
                   "answer": r["answer"], "latency_ms": r["latency_ms"],
                   "llm_calls": r["llm_calls"], "token_usage": r["token_usage"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records[case["id"]] = rec
            totals["gen_calls"] += r["llm_calls"]
            for k in ("prompt_tokens", "completion_tokens"):
                totals[k] += r["token_usage"].get(k, 0)

    # fresh score + judge + attribute
    attrs = []
    per_case_correct: dict[str, bool] = {}
    for cid, case in cases.items():
        rec = records[cid]
        if case["answer_schema"] == "text":
            v = judge_body_text(case["question"], case["expected_answer"]["key_points"], rec["answer"], config, client)
            correct, scored = v["correct"], {"score_detail": {"judge_reason": v["reason"]}}
            totals["judge_calls"] += 1
        else:
            scored = score_case(case, rec["answer"])
            correct = scored["correct"]
        a = attribute(case, correct, scored, rec, config, client)
        a.update(gate_fields(case, rec, a["primary_failure"]))  # self-contained gate input
        attrs.append(a)
        per_case_correct[cid] = a["primary_failure"] == "correct"

    attr_path = Path("reports") / f"attribution_{run_id}.jsonl"
    attr_path.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in attrs) + "\n", encoding="utf-8")
    return M.aggregate(run_id), per_case_correct, totals


def main(config: RagConfig = DEFAULT_CONFIG) -> int:
    from app.env import require_openai_key
    require_openai_key()
    from openai import OpenAI
    client = OpenAI()

    cases = {c["id"]: c for c in (json.loads(l) for l in
             Path("data/eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    pipe = Pipeline(config)
    logger.info("noise band: %d runs, same config sig=%s, temperature=0, seed=%d",
                N_RUNS, config.index_signature(), config.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_sets, stability, run_ids = [], defaultdict(int), []
    grand = defaultdict(float)
    wall0 = time.perf_counter()
    for i in range(1, N_RUNS + 1):
        run_id = f"noise_{stamp}_{config.index_signature()}_r{i}"
        run_ids.append(run_id)
        m, pcc, tot = _one_run(pipe, cases, config, client, run_id)
        metrics_sets.append(m)
        for cid, ok in pcc.items():
            stability[cid] += int(ok)
        for k, v in tot.items():
            grand[k] += v
        logger.info("  run %d/%d (%s): answerable_acc=%s retrieval_miss=%s",
                    i, N_RUNS, run_id, m["answerable_accuracy"], m["failure_distribution"].get("retrieval_miss"))

    elapsed = time.perf_counter() - wall0
    _report_and_save(metrics_sets, stability, cases, run_ids, grand, elapsed, config)
    return 0


def _band(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {"mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals)}


def _report_and_save(metrics_sets, stability, cases, run_ids, grand, elapsed, config):
    band = {k: _band([m.get(k) for m in metrics_sets]) for k in TRACKED}
    for mode in TRACKED_MODES:
        band[f"mode:{mode}"] = _band([m["failure_distribution"].get(mode, 0) for m in metrics_sets])

    unstable = sorted(
        ({"id": cid, "slice": cases[cid]["slice"], "correct_count": c}
         for cid, c in stability.items() if 0 < c < N_RUNS),
        key=lambda x: abs(x["correct_count"] - N_RUNS / 2))  # most split first

    cost = grand["prompt_tokens"] / 1e6 * M._PRICE_IN + grand["completion_tokens"] / 1e6 * M._PRICE_OUT

    logger.info("===== NOISE BAND (n=%d) =====", N_RUNS)
    logger.info("%-34s %8s %7s %5s %5s", "metric", "mean", "std", "min", "max")
    for k, b in band.items():
        logger.info("  %-32s %8s %7s %5s %5s", k, b["mean"], b["std"], b["min"], b["max"])
    logger.info("answerable_accuracy std=%.4f → 약 ±%.1f개/85 흔들림",
                band["answerable_accuracy"]["std"], band["answerable_accuracy"]["std"] * 85)
    logger.info("unstable cases (verdict flipped, 0<k<%d): %d", N_RUNS, len(unstable))
    for u in unstable:
        logger.info("  %s [%s] correct %d/%d", u["id"], u["slice"], u["correct_count"], N_RUNS)
    logger.info("cost ~$%.4f (judge gpt-4o 별도 호출 포함) · wall %.1fs · gen_calls=%d judge_calls=%d",
                cost, elapsed, int(grand["gen_calls"]), int(grand["judge_calls"]))

    OUT_PATH.write_text(json.dumps({
        "n_runs": N_RUNS, "config": config.fingerprint(), "run_ids": run_ids,
        "metric_band": band,
        "case_stability": dict(stability),
        "unstable_cases": unstable,
        "cost_usd_generation": round(cost, 4),
        "wall_seconds": round(elapsed, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
