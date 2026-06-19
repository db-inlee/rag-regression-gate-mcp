"""Quality gate (Phase 3.3): turn a detect report into PASS / WARN / FAIL.

Only statistically-significant regressions FAIL. The report speaks in FAILURE
MODES ("retrieval_miss 65→77 유의 증가 → 검색 회귀"), not raw score drops.
"""

from __future__ import annotations

# Metrics whose significant regression triggers FAIL (per §3.3).
FAIL_METRICS = {"answerable_accuracy", "retrieval_miss", "hallucination", "over_answer_rate"}

# metric -> (human name, regression word, diagnosis)
_PHRASE = {
    "answerable_accuracy": ("정답 정확도(grounded)", "하락", "정확도 회귀"),
    "retrieval_miss": ("retrieval_miss", "증가", "검색 회귀"),
    "hallucination": ("hallucination", "증가", "생성 환각 회귀"),
    "over_answer_rate": ("over_answer_rate", "증가", "과답변 회귀"),
    "over_answer": ("over_answer 건수", "증가", "과답변 회귀"),
    "retrieval_success_strict": ("retrieval_success(strict)", "하락", "검색 성공률 하락"),
    "retrieval_success_value_present": ("retrieval_success(value)", "하락", "검색 성공률 하락"),
    "no_answer_accuracy": ("no_answer 정확도", "하락", "거부 정확도 하락"),
}


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def line(r: dict, significant: bool = True) -> str:
    name, word, diag = _PHRASE.get(r["metric"], (r["metric"], "변화", "변화"))
    ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
    qual = "유의 " if significant else ""
    tail = "" if significant else "  [경계 — CI가 0에 걸치거나 밴드 근처]"
    return (f"{name} {_fmt(r['baseline'])}→{_fmt(r['candidate'])} {qual}{word}"
            f"(Δ{r['delta']:+.4f}, CI {ci}) → {diag}{tail}")


def evaluate(report: dict) -> dict:
    fails, warns, improvements = [], [], []
    for r in report["results"]:
        if r["direction"] == "regression":
            (fails if r["metric"] in FAIL_METRICS else warns).append(r)
        elif r["direction"] == "warn":
            warns.append(r)
        elif r["direction"] == "improvement":
            improvements.append(r)
    status = "FAIL" if fails else ("WARN" if warns else "PASS")
    return {"status": status, "fails": fails, "warns": warns, "improvements": improvements}


def render(report: dict, gate: dict) -> str:
    icon = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}[gate["status"]]
    out = [f"{icon} GATE: {gate['status']}",
           f"  baseline : {report['baseline_run']}",
           f"  candidate: {report['candidate_run']}"]
    if gate["fails"]:
        out.append("  ── 유의한 회귀 (FAIL 사유) — 실패모드 중심 ──")
        for r in sorted(gate["fails"], key=lambda x: x["metric"] != "answerable_accuracy"):
            out.append("   ❌ " + line(r))
    if gate["warns"]:
        out.append("  ── 경계/주의 (WARN, 유의하지 않음) ──")
        for r in gate["warns"]:
            out.append("   ⚠️  " + line(r, significant=False))
    if gate["improvements"]:
        out.append("  ── 유의한 개선 ──")
        for r in gate["improvements"]:
            out.append("   ✅ " + line(r))
    if gate["status"] == "PASS":
        out.append("  유의한 회귀 없음 (노이즈 범위 내 변화).")
    return "\n".join(out)


def exit_code(gate: dict) -> int:
    return 1 if gate["status"] == "FAIL" else 0  # WARN/PASS pass CI (0)
