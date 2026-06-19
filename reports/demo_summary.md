# RAG Regression Guard — 메타 평가 데모 (Phase 5)

동일 baseline에 **config 한 개씩만** 바꿔 candidate를 만들고, 동일 게이트(부트스트랩 유의성 + 노이즈밴드)로 판정. 임계 조작·결과 꾸미기 없음.

- baseline: `noise_20260616_204154_1a13b95a9b06_r1` — answerable_accuracy **0.2**, retrieval_miss **65**
- A 재인덱싱: chunk_size **200→400 완화**(200은 65,647청크/~43분 > 30분 한도)

> **한 줄 요약**: 게이트는 **진짜 회귀(A)는 FAIL**, **헛개선(B)·중립(C)은 FAIL 회피** — 분별 있게 반응한다.

| 변경 | answerable_acc | retrieval_miss | 게이트 | 진단 |
|---|---|---|---|---|
| (baseline) | 0.2 | 65 | — | — |
| A. 검색 회귀 — top_k 5→1 | 0.2→0.0824 | 65→73 | 🔴 FAIL | 검색 회귀 |
| B. reranker off→on | 0.2→0.1882 | 65→60 | 🟡 WARN | 생성 환각 회귀 (경계) |
| C. overlap 150→155 (중립 의도) | 0.2→0.1882 | 65→66 | 🟡 WARN | 정확도 회귀 (경계) |

**CI(재현용)**
- A. 검색 회귀 — top_k 5→1: answerable Δ CI [-0.202, -0.056], retrieval_miss Δ CI [+2.0, +14.0] → 🔴 FAIL
- B. reranker off→on: answerable Δ CI [-0.058, +0.025], retrieval_miss Δ CI [-12.0, +1.0] → 🟡 WARN
- C. overlap 150→155 (중립 의도): answerable Δ CI [-0.043, +0.000], retrieval_miss Δ CI [-2.0, +5.0] → 🟡 WARN
- A-1. chunk_size 1000→400: answerable Δ CI [-0.086, +0.025], retrieval_miss Δ CI [-1.0, +10.0] → 🟡 WARN

## 메시지
1. **진짜 회귀는 잡는다** — A(top_k 5→1)에서 answerable 유의 하락 + retrieval_miss 유의 증가 → 🔴 FAIL, **검색 회귀**로 진단.
2. **중립/무효과엔 FAIL로 짖지 않는다** — B(reranker)·C(overlap)는 유의한 회귀가 아니므로 FAIL 아님(WARN/통과).
3. **점수가 아니라 실패모드로 말한다** — "answerable -12%p"가 아니라 "retrieval_miss 65→73 유의 증가 → 검색 회귀".

## ★ 핵심 발견 — B(reranker)가 '헛개선'이었다 (회귀 가드의 존재 이유)
reranker on은 **직관적으로 검색 품질 개선이 예상**되는 변경이다. 그러나 데이터상 **유의한 개선이 없었고**(answerable Δ CI가 0 포함), 오히려 **hallucination이 경계 증가**했다. 게이트는 이 변경을 '개선'으로 인정하지 않았다(WARN, IMPROVED 아님).
> 이것이 회귀 가드의 존재 이유다: **"좋아질 거라 믿고 바꾼 변경"이 실제로는 개선이 아님을 데이터로 잡는다.** 직관·기대가 아니라 측정으로 판정하므로, 헛개선이 '개선'으로 머지되는 것을 막는다.

## 정직성 노트 (가설과 다른 결과)
- **A-1(chunk_size 1000→400)**: 🟡 WARN — 정확도 0.2→0.1765, CI [-0.086, +0.025](0 포함)으로 **회귀 불충분**. 가설은 FAIL이었으나 데이터는 경계 → **top_k 5→1로 강화**해 FAIL 확보(A 행).
- **B(reranker on)**: 위 핵심 발견 참조. (B는 fetch 20→rerank→top5로 후보풀 확대+reranking **혼합**이라, 순수 reranker 효과 주장은 불가 — 그럼에도 개선 신호 자체가 없음.)
- **C(overlap 150→155, 중립 의도) = WARN인 이유**: overlap의 미세 변경(150→155)도 **config 서명이 바뀌어 재인덱싱**을 유발하고, 청크 경계가 살짝 달라져 **1케이스가 경계에서 흔들렸다**. delta가 floor(±1 case)를 아슬하게 넘고 CI는 0에 닿음(유의 아님) → WARN. **floor(±1 case)가 FAIL 오판은 막았다**(중립을 회귀로 잘못 차단하지 않음, exit 0). 완전 PASS(침묵)를 원하면 **floor를 ±2 case로 조정 가능** — 트레이드오프는 **민감도 하락**(2케이스 이내의 진짜 작은 회귀를 놓칠 수 있음). 이번 데모에선 임계를 손대지 않고 측정 그대로 보고.

> 게이트 exit code: FAIL=1(A), 그 외=0(B·C). suggestion-only — 자동 수정 없음. 예: A 회귀 시 "top_k 되돌림 검토" 제안만.
