<!--
이 파일은 GitHub Actions에서 게이트가 FAIL했을 때 PR에 게시되는 코멘트의 예시다.
scripts/run_gate.py가 gate_runs/gate_<candidate>.json을 만들고, CI가 그걸 마크다운으로 렌더해
PR에 코멘트한다. 아래 수치는 실제 examples/allganize_candidate(top_k 5→1) gate_result 기반이다.
-->

## 🔴 RAG Regression Gate: **FAIL**

`examples/allganize_baseline` → `examples/allganize_candidate` 비교 결과, **유의한 품질 회귀**가 감지되어 머지가 차단됩니다.

> ⛔ **이 변경은 머지가 차단됩니다 (`exit 1`).** 아래 회귀를 해소하거나, 의도된 변경이면 baseline을 갱신하세요.

### 회귀한 메트릭 (유의 — FAIL 사유)

| 메트릭 | baseline → candidate | Δ (cases) | 95% CI | 판정 |
|---|---|---|---|---|
| **answerable_accuracy** (grounded) | 0.55 → 0.30 | −0.25 (10.0 cases) | [−0.40, −0.125] | 🔴 정확도 회귀 |
| **retrieval_miss** | 1 → 5 | +4 (4 cases) | [+1, +8] | 🔴 검색 회귀 |
| retrieval_success (strict) | 0.975 → 0.875 | −0.10 (4.0 cases) | [−0.20, −0.025] | 🔴 검색 성공률 하락 |
| retrieval_success (value) | 0.975 → 0.875 | −0.10 (4.0 cases) | [−0.20, −0.025] | 🔴 검색 성공률 하락 |

판정 규칙: **부트스트랩 CI가 0을 제외**(유의) **AND** **|Δ| > 노이즈밴드 floor**(`effective_band_cases = 1.0`, 모집단 40 기준)일 때만 회귀. 두 조건을 모두 넘었습니다.

### 경계 신호 (WARN — 참고)

| 메트릭 | baseline → candidate | Δ | 95% CI |
|---|---|---|---|
| hallucination | 17 → 23 | +6 | [0, +12] *(CI가 0에 걸침)* |

### 원인 후보 & 제안 (suggestion-only)

- **원인 config 변경**: `top_k` **5 → 1** (검색 약화).
- 🔧 **[retrieval_miss] 검색 단계 회귀** — 우선 **되돌림(`top_k` 1 → 5)** 검토. 그래도 부족하면: top_k 상향, 청크 크기 축소(과대청크 분할), `chunk_overlap` 증가.
- 🔧 **[hallucination] 그라운딩 단계** — (candidate config에 직접 원인 파라미터는 안 보임) 근거 인용 강제(citation), 컨텍스트 밖 지식 금지 프롬프트 강화, 근거 부족 시 abstain 유도.

> 제안은 **검토 후보**이며 자동 적용되지 않습니다(룰 기반 카탈로그, LLM 미사용). 적용 후 이 게이트로 재검증하세요.

---
<sub>RAG Regression Gate · 입출력 스키마: [`docs/artifact_contract.md`](../docs/artifact_contract.md) · 판정 근거: [`docs/adr/ADR-002`](../docs/adr/ADR-002-bootstrap-ci-noise-band.md) · 같은 수치를 CLI/MCP/REST 어디서나 (5중 일치).</sub>
