"""Regression detection via paired bootstrap (Phase 3.2).

Two runs share the SAME 100 cases, so we resample case indices (paired): one
resample is used to recompute BOTH baseline and candidate metrics, and we take
their difference. ~1000 resamples give the 95% CI of the delta.

A change is a SIGNIFICANT regression only when BOTH hold:
  (a) the bootstrap CI excludes 0 in the regression direction (sampling uncertainty)
  (b) |delta| exceeds the effective noise band (run-to-run nondeterminism)

effective_band = max(measured_band, 1 case): the measured band stays honest (0
when fully deterministic) in noise_band.json; the +1-case floor is applied HERE so
a single flaky case is never called a real regression.

If only one of (a)/(b) holds → WARN.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.evaluator.attribution import is_retrieval_miss
from app.evaluator.metrics import value_present

RUNS_DIR = Path("reports/runs")
NOISE_BAND = Path("reports/noise_band.json")
BOOT = 1000

# metric -> (kind, population, denom, regression_direction, noise_key)
#   kind: "prop" (rate over denom) | "count" (cases out of population)
#   regression_direction: "down" (lower=worse) | "up" (higher=worse)
METRICS = {
    "answerable_accuracy":            ("prop", "answerable", 85, "down", "answerable_accuracy"),
    "retrieval_success_strict":       ("prop", "answerable", 85, "down", "retrieval_success_strict"),
    "retrieval_success_value_present":("prop", "answerable", 85, "down", "retrieval_success_value_present"),
    "no_answer_accuracy":             ("prop", "no_answer", 15, "down", "no_answer_accuracy"),
    "over_answer_rate":               ("prop", "no_answer", 15, "up",   "over_answer_rate"),
    "retrieval_miss":                 ("count", "all", 100, "up", "mode:retrieval_miss"),
    "hallucination":                  ("count", "all", 100, "up", "mode:hallucination"),
    "over_answer":                    ("count", "no_answer", 15, "up", "over_answer_rate"),
}


def _load_cases() -> dict:
    return {c["id"]: c for c in (json.loads(l) for l in
            Path("data/eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}


def case_features(run_id: str, cases: dict) -> dict:
    """Per-case booleans for one run (from its log + attribution)."""
    recs = {r["id"]: r for r in (json.loads(l) for l in
            (RUNS_DIR / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
            if r.get("type") == "case"}
    attr = {a["id"]: a["primary_failure"] for a in (json.loads(l) for l in
            (Path("reports") / f"attribution_{run_id}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    feats = {}
    for cid, case in cases.items():
        rec = recs.get(cid, {})
        pf = attr.get(cid, "")
        correct = pf == "correct"
        answerable = case["answer_type"] == "answerable"
        vp = value_present(case, rec)
        feats[cid] = {
            "answerable": answerable,
            "grounded_correct": answerable and correct and vp,
            "strict_ok": answerable and not is_retrieval_miss(case, rec),
            "value_ok": answerable and vp,
            "na_correct": (not answerable) and correct,
            "over_answer": (not answerable) and (not correct),
            "mode": pf,
        }
    return feats


def _metric_value(ids: list[str], feats: dict, metric: str) -> float | None:
    kind, pop, _, _, _ = METRICS[metric]
    if pop == "answerable":
        sub = [i for i in ids if feats[i]["answerable"]]
    elif pop == "no_answer":
        sub = [i for i in ids if not feats[i]["answerable"]]
    else:
        sub = ids
    if not sub:
        return None
    if metric == "answerable_accuracy":
        return sum(feats[i]["grounded_correct"] for i in sub) / len(sub)
    if metric == "retrieval_success_strict":
        return sum(feats[i]["strict_ok"] for i in sub) / len(sub)
    if metric == "retrieval_success_value_present":
        return sum(feats[i]["value_ok"] for i in sub) / len(sub)
    if metric == "no_answer_accuracy":
        return sum(feats[i]["na_correct"] for i in sub) / len(sub)
    if metric == "over_answer_rate":
        return sum(feats[i]["over_answer"] for i in sub) / len(sub)
    if metric == "retrieval_miss":
        return sum(feats[i]["mode"] == "retrieval_miss" for i in sub)
    if metric == "hallucination":
        return sum(feats[i]["mode"] == "hallucination" for i in sub)
    if metric == "over_answer":
        return sum(feats[i]["over_answer"] for i in sub)
    return None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _effective_band_cases(metric: str, band_file: dict) -> float:
    """Measured noise band converted to CASES, floored at 1."""
    _, _, denom, _, key = METRICS[metric]
    std = (band_file.get("metric_band", {}).get(key, {}) or {}).get("std") or 0.0
    kind = METRICS[metric][0]
    band_cases = std * denom if kind == "prop" else std
    return max(band_cases, 1.0)


def features_from_enriched(attr_path: Path) -> dict:
    """Gate-input features straight from an ENRICHED attribution artifact (no gold).

    Requires the per-case fields written by app.evaluator.case_eval.gate_fields.
    """
    feats = {}
    for a in (json.loads(l) for l in attr_path.read_text(encoding="utf-8").splitlines() if l.strip()):
        ans = a["answerable"]
        feats[a["id"]] = {
            "answerable": ans,
            "grounded_correct": ans and a["correct"] and a["value_present"],
            "strict_ok": a["retrieval_strict_ok"],
            "value_ok": ans and a["value_present"],
            "na_correct": (not ans) and a["correct"],
            "over_answer": a.get("over_answer", (not ans) and (not a["correct"])),
            "mode": a["primary_failure"],
        }
    return feats


def _bootstrap(case_ids: list[str], B: dict, C: dict, band_file: dict, config: RagConfig) -> list[dict]:
    rng = random.Random(config.seed)
    results = []
    for metric, (kind, pop, denom, reg_dir, _) in METRICS.items():
        base = _metric_value(case_ids, B, metric)
        cand = _metric_value(case_ids, C, metric)
        delta = (cand - base) if (base is not None and cand is not None) else 0.0

        deltas = []
        for _ in range(BOOT):
            sample = [rng.choice(case_ids) for _ in case_ids]
            b = _metric_value(sample, B, metric)
            c = _metric_value(sample, C, metric)
            if b is not None and c is not None:
                deltas.append(c - b)
        deltas.sort()
        ci_low, ci_high = _percentile(deltas, 0.025), _percentile(deltas, 0.975)

        # (a) CI excludes 0 in regression direction
        ci_excludes_0 = ci_low > 0 or ci_high < 0
        regressing = (delta < 0) if reg_dir == "down" else (delta > 0)
        improving = not regressing and ci_excludes_0
        # (b) |delta| beyond effective band (in cases)
        eff_band = _effective_band_cases(metric, band_file)
        delta_cases = abs(delta) * (denom if kind == "prop" else 1)
        beyond_band = delta_cases > eff_band

        if ci_excludes_0 and regressing and beyond_band:
            status = "regression"
        elif ci_excludes_0 and improving and beyond_band:
            status = "improvement"
        elif (ci_excludes_0 and regressing) or (regressing and beyond_band):
            status = "warn"  # only one condition (regression-leaning)
        else:
            status = "no_change"

        results.append({
            "metric": metric, "kind": kind, "regression_dir": reg_dir,
            "baseline": round(base, 4) if base is not None else None,
            "candidate": round(cand, 4) if cand is not None else None,
            "delta": round(delta, 4), "delta_cases": round(delta_cases, 2),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "effective_band_cases": round(eff_band, 3),
            "significant": status in ("regression", "improvement"),
            "direction": status,
        })
    return results


def detect(baseline_run: str, candidate_run: str, config: RagConfig = DEFAULT_CONFIG) -> dict:
    """Gold mode (local/back-compat): resolve run_ids in reports/, recompute from gold."""
    cases = _load_cases()
    band_file = json.loads(NOISE_BAND.read_text(encoding="utf-8")) if NOISE_BAND.exists() else {}
    B = case_features(baseline_run, cases)
    C = case_features(candidate_run, cases)
    results = _bootstrap(list(cases), B, C, band_file, config)
    return {"baseline_run": baseline_run, "candidate_run": candidate_run, "results": results}


def detect_paths(baseline_dir: Path, candidate_dir: Path, config: RagConfig = DEFAULT_CONFIG) -> dict:
    """CI mode (gold-free): read enriched attribution from each dir. No eval_cases."""
    B = features_from_enriched(baseline_dir / "attribution.jsonl")
    C = features_from_enriched(candidate_dir / "attribution.jsonl")
    band_path = baseline_dir / "noise_band.json"
    band_file = json.loads(band_path.read_text(encoding="utf-8")) if band_path.exists() else {}
    case_ids = sorted(set(B) & set(C))
    results = _bootstrap(case_ids, B, C, band_file, config)
    return {"baseline_run": str(baseline_dir), "candidate_run": str(candidate_dir), "results": results}


def _print(report: dict) -> None:
    print(f"\n===== REGRESSION DETECT =====\nbaseline: {report['baseline_run']}\ncandidate: {report['candidate_run']}")
    print(f"{'metric':32} {'base':>8} {'cand':>8} {'delta':>8} {'95% CI':>18} {'band':>5}  status")
    for r in report["results"]:
        ci = f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
        print(f"  {r['metric']:30} {str(r['baseline']):>8} {str(r['candidate']):>8} "
              f"{r['delta']:+8.4f} {ci:>18} {r['effective_band_cases']:>5}  {r['direction']}")
    regs = [r["metric"] for r in report["results"] if r["direction"] == "regression"]
    print(f"\n유의한 회귀: {len(regs)}건  {regs}")


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2:
        baseline_run, candidate_run = args[0], args[1]
    else:  # default: false-alarm test on the first two noise runs (same config)
        band = json.loads(NOISE_BAND.read_text(encoding="utf-8"))
        baseline_run, candidate_run = band["run_ids"][0], band["run_ids"][1]
        print(f"[거짓경보 테스트] 같은 config 두 실행: {baseline_run} vs {candidate_run}")
    report = detect(baseline_run, candidate_run)
    out = Path("gate_runs") / f"regression_{candidate_run}.json"  # byproduct, not reports/
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
