"""Single-run failure analysis (analyze_failures). THIN AGGREGATOR — no statistics.

Pair to run_gate: run_gate compares TWO runs (regression detection); this diagnoses
ONE run ("where is it weak now, what to fix first?"). No bootstrap/CI here — a single
run has nothing to compare against.

Gold-free by contract (mirrors run_gate, honours Phase 6): reads ONLY the precomputed
attribution.jsonl (per-case booleans: answerable/correct/value_present/
retrieval_strict_ok/over_answer/primary_failure). It does NOT call metrics.aggregate()
— that re-reads eval_cases.jsonl (gold), which would undo Phase 6's gold removal.
So this is plain counting of attribution fields (~deterministic, judge-free), not a
re-implementation of any statistic.

RAGAS equivalents are deterministic conversions of our existing measures (3 of 4):
context_precision and answer_relevancy are intentionally omitted (see RagasEquivalent.note).

suggestion-only: improvement_priorities are review candidates, never auto-applied;
each carries the closed-loop footer ("apply, then re-verify with run_gate").
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.suggest import _STAGE_KR, _parse_catalog

CLOSED_LOOP = "(자동 적용 안 함. 적용 후 run_gate로 개선 여부 검증.)"


class SliceStat(BaseModel):
    """Per-slice failure rate + its most common failure mode."""

    slice: str
    total: int
    failed: int
    fail_rate: float
    dominant_failure: str


class RagasEquivalent(BaseModel):
    """RAGAS concepts converted from our deterministic measures (judge-free, reproducible)."""

    context_recall: float | None        # ≈ retrieval_success_strict (gold 근거 ⊆ retrieved)
    faithfulness: float | None          # ≈ grounded / (grounded + unsupported)
    answer_correctness: float | None    # ≈ answerable_accuracy (grounded 기준)
    note: str


class FailureAnalysis(BaseModel):
    """Single run-log diagnosis: distribution + bottleneck + slices + groundedness +
    RAGAS-equivalent + improvement priorities. All numbers deterministic (no judge)."""

    n_cases: int
    failure_distribution: dict[str, int]   # Counter(primary_failure) — matches metrics.py
    bottleneck_stage: str                  # retrieval / grounding / refusal / generation_reasoning / none
    bottleneck_reason: str
    slice_stats: list[SliceStat]
    grounded_correct: int
    unsupported_correct: int               # 맞았지만 근거 미실재 = 잠재 리스크
    groundedness_note: str
    ragas_equivalent: RagasEquivalent
    improvement_priorities: list[str] = Field(default_factory=list)  # suggestion-only


def _load_attr(run_dir: str | Path) -> list[dict]:
    """Read the precomputed attribution.jsonl (the only input — gold-free)."""
    path = Path(run_dir) / "attribution.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _round(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def build_analysis(run_dir: str) -> FailureAnalysis:
    """Aggregate one run's attribution.jsonl into a deterministic diagnosis."""
    records = _load_attr(run_dir)
    cat = _parse_catalog()

    dist: Counter = Counter()
    by_slice: dict[str, dict] = defaultdict(lambda: {"total": 0, "failed": 0, "modes": Counter()})
    ans_total = grounded = unsupported = rs_strict = 0
    na_total = over = 0

    for a in records:
        pf = a["primary_failure"]
        dist[pf] += 1
        sl = by_slice[a["slice"]]
        sl["total"] += 1
        if pf != "correct":
            sl["failed"] += 1
            sl["modes"][pf] += 1
        if a.get("answerable"):
            ans_total += 1
            if a.get("retrieval_strict_ok"):
                rs_strict += 1
            if a.get("correct"):
                if a.get("value_present"):
                    grounded += 1
                else:
                    unsupported += 1
        else:
            na_total += 1
            if a.get("over_answer"):
                over += 1

    # --- bottleneck: dominant non-correct failure mode → catalog stage ---------- #
    non_correct = [(m, n) for m, n in dist.items() if m != "correct"]
    if non_correct:
        mode, cnt = max(non_correct, key=lambda x: x[1])
        stage = cat["techniques"].get(mode, {}).get("stage", "unknown")
        stage_kr = _STAGE_KR.get(stage, stage)
        bottleneck_stage = stage
        bottleneck_reason = f"{mode}가 {cnt}건으로 가장 큰 병목 ({stage_kr} 단계)"
    else:
        mode, cnt, stage = None, 0, "none"
        bottleneck_stage = "none"
        bottleneck_reason = "실패 케이스 없음 (전부 correct)"

    # --- slice stats ------------------------------------------------------------ #
    slice_stats = [
        SliceStat(
            slice=s, total=v["total"], failed=v["failed"],
            fail_rate=round(v["failed"] / v["total"], 4) if v["total"] else 0.0,
            dominant_failure=v["modes"].most_common(1)[0][0] if v["modes"] else "correct",
        )
        for s, v in sorted(by_slice.items(), key=lambda kv: -kv[1]["failed"])
    ]

    # --- groundedness ----------------------------------------------------------- #
    groundedness_note = (
        f"정답(grounded {grounded} + 암기 {unsupported}) 중 {unsupported}건은 근거 미실재"
        f"(unsupported) = 잠재 리스크 — 검색이 좋아지면 안정화."
        if unsupported else "맞은 답은 모두 근거 실재(unsupported 정답 0건)."
    )

    # --- RAGAS equivalents (3, deterministic) ----------------------------------- #
    ragas = RagasEquivalent(
        context_recall=_round(rs_strict, ans_total),
        faithfulness=_round(grounded, grounded + unsupported),
        answer_correctness=_round(grounded, ans_total),
        note=(
            "RAGAS 개념을 우리의 결정적 측정으로 환산 — LLM judge 미사용이라 같은 입력엔 같은 값(재현 가능). "
            "context_recall≈retrieval_success_strict(gold 근거 ⊆ retrieved), "
            "faithfulness≈grounded/(grounded+unsupported), "
            "answer_correctness≈answerable_accuracy(grounded 기준). "
            "context_precision은 precomputed attribution(gold-free)만으론 산출 불가 — "
            "gold-free 계약 유지를 위해 의도적 생략. "
            "answer_relevancy도 judge 필요(결정성과 충돌)라 의도적 제외."
        ),
    )

    # --- improvement priorities (A3, suggestion-only, catalog-driven) ----------- #
    priorities: list[str] = []
    if mode:  # 1) bottleneck-stage techniques for the dominant mode
        entry = cat["techniques"].get(mode)
        if entry:
            techs = "; ".join(entry["techs"][:3])
            stage_kr = _STAGE_KR.get(entry["stage"], entry["stage"])
            priorities.append(
                f"우선순위 {len(priorities)+1}: [{stage_kr} 단계 — {mode} {cnt}건] "
                f"→ {techs} (카탈로그). {CLOSED_LOOP}"
            )
    # 2) slice concentration: high-fail slices with a catalog emphasis
    for st in slice_stats:
        if st.fail_rate >= 0.5 and st.total >= 5 and st.slice in cat["slice_emphasis"]:
            priorities.append(
                f"우선순위 {len(priorities)+1}: [슬라이스 집중 — {st.slice} "
                f"{st.failed}/{st.total} 실패] → {cat['slice_emphasis'][st.slice]} (카탈로그). {CLOSED_LOOP}"
            )
    # 3) groundedness risk: unsupported corrects are "맞지만 위태로운 정답" → 검색 강화
    if unsupported >= 2:
        priorities.append(
            f"우선순위 {len(priorities)+1}: [groundedness 리스크 — unsupported {unsupported}건] "
            f"→ 검색 개선으로 근거 확보(맞은 답을 근거 실재로 안정화). {CLOSED_LOOP}"
        )

    return FailureAnalysis(
        n_cases=len(records),
        failure_distribution=dict(dist),
        bottleneck_stage=bottleneck_stage,
        bottleneck_reason=bottleneck_reason,
        slice_stats=slice_stats,
        grounded_correct=grounded,
        unsupported_correct=unsupported,
        groundedness_note=groundedness_note,
        ragas_equivalent=ragas,
        improvement_priorities=priorities,
    )
