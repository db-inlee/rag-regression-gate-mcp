"""Validate judge reliability against gold (Phase 2.2) — the differentiator.

For each body_text gold case we build three probes and check the judge's verdict:
  (a) positive  : the gold answer itself            -> judge must say correct
  (b-i) negative: another paragraph from the SAME report (plausible, off-topic)
  (b-ii) negative: a SUBTLE alteration of the gold (one fact changed)
                   -> judge must say incorrect

Subtle (b-ii) distractors are crafted by the generation model so the judge is
tested on hard, plausible wrongs — not absurd ones that inflate accuracy.

Reports judge_accuracy + confusion matrix; flags '신뢰 불가' if accuracy < 0.85.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.env import require_openai_key
from app.evaluator.judge import judge_body_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("validate_judge")

EVAL_PATH = Path("data/eval_cases.jsonl")
OUT_PATH = Path("reports/judge_validation.json")
THRESHOLD = 0.85


def _craft_subtle(client, key_points: list[str], config: RagConfig, temperature: float) -> str:
    """Craft a distractor that CHANGES a core fact to a false one (not additive)."""
    resp = client.chat.completions.create(
        model=config.generation_model, temperature=temperature, seed=config.seed,
        messages=[{"role": "user", "content": (
            "다음 정답의 핵심 사실 하나를 '틀린 값'으로 바꿔라.\n"
            "허용: 용어 교체(예: 재분류→재조정), 핵심 숫자를 다른 숫자로, 핵심 항목/대상을 다른 것으로.\n"
            "금지: 옳은 내용을 덧붙이기, 표현만 바꾸고 사실은 유지하기 — 결과는 원래와 사실이 "
            "'명백히 어긋나야' 한다. 길이·형식은 원래와 비슷하게. 바뀐 답 문장만 출력하라.\n"
            "정답: " + " ".join(key_points))}],
    )
    return (resp.choices[0].message.content or "").strip()


def _contradicts_gold(client, key_points: list[str], variation: str, config: RagConfig) -> bool:
    """Self-check: does the variation factually conflict with the gold key_points?"""
    resp = client.chat.completions.create(
        model=config.judge_model, temperature=0, seed=config.seed,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": (
            "원래 정답 핵심 포인트와 '변형 답'을 비교하라. 변형 답이 핵심 사실(용어·숫자·항목·방향)을 "
            "틀리게 바꿔 원래와 사실이 어긋나면 contradicts=true. 단순히 옳은 내용을 덧붙였거나 "
            "표현만 다르고 여전히 사실과 일치하면 contradicts=false.\n"
            f"핵심 포인트: {key_points}\n변형 답: {variation}\n"
            'JSON: {"contradicts": true/false}')}],
    )
    try:
        return bool(json.loads(resp.choices[0].message.content).get("contradicts"))
    except Exception:  # noqa: BLE001
        return False


def _make_subtle_wrong(client, key_points: list[str], config: RagConfig) -> tuple[str, bool]:
    """Return (variation, valid). valid=True only if it genuinely contradicts gold.
    Retries with rising temperature; gives up (valid=False) after 3 tries."""
    text = ""
    for temp in (0.0, 0.5, 0.9):
        text = _craft_subtle(client, key_points, config, temp)
        if _contradicts_gold(client, key_points, text, config):
            return text, True
    return text, False


def main(config: RagConfig = DEFAULT_CONFIG) -> int:
    require_openai_key()
    from openai import OpenAI
    client = OpenAI()

    cases = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    body = [c for c in cases if c["slice"] == "body_text"]
    by_report: dict[tuple, list[dict]] = defaultdict(list)
    for c in body:
        by_report[(c["company"], c["fiscal_year"])].append(c)

    def other_same_report(case: dict) -> dict:
        peers = [c for c in by_report[(case["company"], case["fiscal_year"])] if c["id"] != case["id"]]
        pool = peers or [c for c in body if c["id"] != case["id"]]
        return pool[0]

    logger.info("validating judge_model=%s on %d body_text gold cases (×3 probes)",
                config.judge_model, len(body))

    probes: list[dict] = []
    invalid_subtle: list[str] = []
    for case in body:
        q, kp = case["question"], case["expected_answer"]["key_points"]
        gold_answer = " ".join(kp)
        neg_i = " ".join(other_same_report(case)["expected_answer"]["key_points"])
        neg_ii, valid = _make_subtle_wrong(client, kp, config)
        probes.append({"id": case["id"], "kind": "pos", "expected": True, "q": q, "kp": kp, "ans": gold_answer})
        probes.append({"id": case["id"], "kind": "neg_diff_paragraph", "expected": False, "q": q, "kp": kp, "ans": neg_i})
        if valid:  # only genuinely-wrong variations are valid probes
            probes.append({"id": case["id"], "kind": "neg_subtle", "expected": False, "q": q, "kp": kp, "ans": neg_ii})
        else:
            invalid_subtle.append(case["id"])
    if invalid_subtle:
        logger.warning("dropped %d invalid neg_subtle probes (not genuinely wrong): %s",
                       len(invalid_subtle), invalid_subtle)

    parse_fail = 0
    results = []
    for p in probes:
        v = judge_body_text(p["q"], p["kp"], p["ans"], config, client)
        parse_fail += 0 if v["parse_ok"] else 1
        results.append({**p, "judge_correct": v["correct"], "parse_ok": v["parse_ok"],
                        "judge_version": v["judge_version"]})

    _report(results, parse_fail, config, invalid_subtle)
    return 0


def _report(results: list[dict], parse_fail: int, config: RagConfig,
            invalid_subtle: list[str] | None = None) -> None:
    tp = sum(r["judge_correct"] for r in results if r["expected"])
    pos = sum(r["expected"] for r in results)
    fn = pos - tp
    tn = sum(not r["judge_correct"] for r in results if not r["expected"])
    neg = sum(not r["expected"] for r in results)
    fp = neg - tn
    total = len(results)
    accuracy = (tp + tn) / total if total else 0.0

    # per negative type (which distractors fool the judge)
    by_kind = defaultdict(lambda: [0, 0])  # kind -> [correct_verdict, total]
    for r in results:
        ok = r["judge_correct"] == r["expected"]
        by_kind[r["kind"]][0] += ok
        by_kind[r["kind"]][1] += 1

    logger.info("===== JUDGE VALIDATION =====")
    version = results[0].get("judge_version") if results else None
    logger.info("judge_model: %s | version: %s", config.judge_model, version)
    logger.info("probes: %d  (parse failures: %d, dropped invalid neg_subtle: %d %s)",
                total, parse_fail, len(invalid_subtle or []), invalid_subtle or [])
    logger.info("confusion matrix:")
    logger.info("                 judge=correct   judge=incorrect")
    logger.info("  gold=correct        TP=%-3d          FN=%-3d", tp, fn)
    logger.info("  gold=incorrect      FP=%-3d          TN=%-3d", fp, tn)
    logger.info("per probe type (정답판정/전체):")
    for kind, (ok, n) in by_kind.items():
        logger.info("  %-20s %d/%d", kind, ok, n)
    logger.info("judge_accuracy = %.3f", accuracy)
    if accuracy < THRESHOLD:
        logger.warning("⚠️  judge_accuracy %.3f < %.2f → judge '신뢰 불가' (리포트에 경고 표기)",
                       accuracy, THRESHOLD)
    else:
        logger.info("judge_accuracy ≥ %.2f → 신뢰 가능", THRESHOLD)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "judge_model": config.judge_model,
        "judge_accuracy": round(accuracy, 4),
        "trustworthy": accuracy >= THRESHOLD,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "by_probe_kind": {k: {"correct": v[0], "total": v[1]} for k, v in by_kind.items()},
        "parse_failures": parse_fail,
        "invalid_subtle_dropped": invalid_subtle or [],
        "n_probes": total,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
