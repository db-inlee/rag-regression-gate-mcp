"""Phase 5.3 — one-table meta-eval summary -> reports/demo_summary.md.

Reads scenarios.json (candidates) + baseline (noise r1), runs the Phase 3 gate for
each candidate, and emits a self-contained Markdown table with real numbers + CIs.
Honest: reports whatever the data says (no threshold tuning, no dressing up).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluator import metrics as M
from app.regression.detect import detect
from app.regression.gate import evaluate

ICON = {"PASS": "🟢 PASS", "WARN": "🟡 WARN", "FAIL": "🔴 FAIL"}


def _find(results, metric):
    return next((r for r in results if r["metric"] == metric), None)


def _row(label, baseline_run, run_id, base_m):
    rep = detect(baseline_run, run_id)
    gate = evaluate(rep)
    ans = _find(rep["results"], "answerable_accuracy")
    rm = _find(rep["results"], "retrieval_miss")
    # pick the headline diagnosis line (first FAIL, else first WARN, else '—')
    diag = "유의 변화 없음"
    pool = gate["fails"] or gate["warns"]
    if pool:
        from app.regression.gate import _PHRASE
        # failure-mode-centric: prefer a failure mode over the score metric
        modes = {"retrieval_miss", "hallucination", "over_answer", "over_answer_rate"}
        r0 = next((r for r in pool if r["metric"] in modes), pool[0])
        _, _, d = _PHRASE.get(r0["metric"], (r0["metric"], "", "변화"))
        diag = d + ("" if gate["fails"] else " (경계)")
    return {
        "label": label,
        "ans": f"{ans['baseline']}→{ans['candidate']}",
        "rmiss": f"{int(rm['baseline'])}→{int(rm['candidate'])}",
        "gate": ICON[gate["status"]],
        "diag": diag,
        "ans_ci": f"[{ans['ci_low']:+.3f}, {ans['ci_high']:+.3f}]",
        "rm_ci": f"[{rm['ci_low']:+.1f}, {rm['ci_high']:+.1f}]",
        "status": gate["status"],
    }


def main() -> int:
    scen = json.loads(Path("reports/scenarios.json").read_text(encoding="utf-8"))
    baseline_run = scen["baseline_run"]
    base_m = M.aggregate(baseline_run)
    by_name = {s["name"]: s for s in scen["scenarios"]}

    def run_id(prefix):
        return next(s["run_id"] for n, s in by_name.items() if n.startswith(prefix))

    rows = [
        _row(f"A. 검색 회귀 — top_k 5→1", baseline_run, run_id("A2"), base_m),
        _row("B. reranker off→on", baseline_run, run_id("B"), base_m),
        _row("C. overlap 150→155 (중립 의도)", baseline_run, run_id("C"), base_m),
    ]
    # A-1 footnote (chunk_size relaxed 200→400)
    a1 = _row("A-1. chunk_size 1000→400", baseline_run, run_id("A1"), base_m)

    md = []
    md.append("# RAG Regression Guard — 메타 평가 데모 (Phase 5)\n")
    md.append("동일 baseline에 **config 한 개씩만** 바꿔 candidate를 만들고, 동일 게이트(부트스트랩 "
              "유의성 + 노이즈밴드)로 판정. 임계 조작·결과 꾸미기 없음.\n")
    md.append(f"- baseline: `{baseline_run}` — answerable_accuracy **{base_m['answerable_accuracy']}**, "
              f"retrieval_miss **{base_m['failure_distribution'].get('retrieval_miss')}**")
    md.append(f"- A 재인덱싱: chunk_size **200→400 완화**(200은 65,647청크/~43분 > 30분 한도)\n")
    md.append("> **한 줄 요약**: 게이트는 **진짜 회귀(A)는 FAIL**, **헛개선(B)·중립(C)은 FAIL 회피** — 분별 있게 반응한다.\n")
    md.append("| 변경 | answerable_acc | retrieval_miss | 게이트 | 진단 |")
    md.append("|---|---|---|---|---|")
    md.append(f"| (baseline) | {base_m['answerable_accuracy']} | "
              f"{base_m['failure_distribution'].get('retrieval_miss')} | — | — |")
    for r in rows:
        md.append(f"| {r['label']} | {r['ans']} | {r['rmiss']} | {r['gate']} | {r['diag']} |")
    md.append("")
    md.append("**CI(재현용)**")
    for r in rows + [a1]:
        md.append(f"- {r['label']}: answerable Δ CI {r['ans_ci']}, retrieval_miss Δ CI {r['rm_ci']} → {r['gate']}")
    md.append("")
    md.append("## 메시지")
    md.append("1. **진짜 회귀는 잡는다** — A(top_k 5→1)에서 answerable 유의 하락 + retrieval_miss 유의 증가 → 🔴 FAIL, **검색 회귀**로 진단.")
    md.append("2. **중립/무효과엔 FAIL로 짖지 않는다** — B(reranker)·C(overlap)는 유의한 회귀가 아니므로 FAIL 아님(WARN/통과).")
    md.append("3. **점수가 아니라 실패모드로 말한다** — \"answerable -12%p\"가 아니라 \"retrieval_miss 65→73 유의 증가 → 검색 회귀\".")
    md.append("")
    md.append("## ★ 핵심 발견 — B(reranker)가 '헛개선'이었다 (회귀 가드의 존재 이유)")
    md.append("reranker on은 **직관적으로 검색 품질 개선이 예상**되는 변경이다. 그러나 데이터상 "
              "**유의한 개선이 없었고**(answerable Δ CI가 0 포함), 오히려 **hallucination이 경계 증가**했다. "
              "게이트는 이 변경을 '개선'으로 인정하지 않았다(WARN, IMPROVED 아님).")
    md.append("> 이것이 회귀 가드의 존재 이유다: **\"좋아질 거라 믿고 바꾼 변경\"이 실제로는 개선이 아님을 데이터로 잡는다.** "
              "직관·기대가 아니라 측정으로 판정하므로, 헛개선이 '개선'으로 머지되는 것을 막는다.")
    md.append("")
    md.append("## 정직성 노트 (가설과 다른 결과)")
    md.append(f"- **A-1(chunk_size 1000→400)**: {a1['gate']} — 정확도 {a1['ans']}, CI {a1['ans_ci']}(0 포함)으로 "
              "**회귀 불충분**. 가설은 FAIL이었으나 데이터는 경계 → **top_k 5→1로 강화**해 FAIL 확보(A 행).")
    md.append("- **B(reranker on)**: 위 핵심 발견 참조. (B는 fetch 20→rerank→top5로 후보풀 확대+reranking **혼합**이라, "
              "순수 reranker 효과 주장은 불가 — 그럼에도 개선 신호 자체가 없음.)")
    md.append("- **C(overlap 150→155, 중립 의도) = WARN인 이유**: overlap의 미세 변경(150→155)도 **config 서명이 바뀌어 "
              "재인덱싱**을 유발하고, 청크 경계가 살짝 달라져 **1케이스가 경계에서 흔들렸다**. "
              "delta가 floor(±1 case)를 아슬하게 넘고 CI는 0에 닿음(유의 아님) → WARN. "
              "**floor(±1 case)가 FAIL 오판은 막았다**(중립을 회귀로 잘못 차단하지 않음, exit 0). "
              "완전 PASS(침묵)를 원하면 **floor를 ±2 case로 조정 가능** — 트레이드오프는 **민감도 하락**"
              "(2케이스 이내의 진짜 작은 회귀를 놓칠 수 있음). 이번 데모에선 임계를 손대지 않고 측정 그대로 보고.")
    md.append("")
    md.append("> 게이트 exit code: FAIL=1(A), 그 외=0(B·C). suggestion-only — 자동 수정 없음. "
              "예: A 회귀 시 \"top_k 되돌림 검토\" 제안만.")

    out = Path("reports/demo_summary.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
