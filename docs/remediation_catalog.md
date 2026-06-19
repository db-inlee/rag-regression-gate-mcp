# 처방 카탈로그 (remediation catalog) — 실패모드 → 단계 → 기법

> M2 제안 엔진(`app/mcp/suggest.py`)이 **근거로 파싱하는 단일 진실원천**이다.
> 제안은 룰 기반·결정적 — LLM이 만들지 않는다. "왜 이 제안인가"는 항상
> **실패모드 → 파이프라인 단계 → 기법**으로 추적된다.
>
> ⚠️ **suggestion-only**(CLAUDE.md §1): 아래 기법은 *검토 후보*이지 정답이 아니다.
> 엔진은 절대 자동 적용/실행하지 않는다. 적용은 사람이 하고, 적용 후 이 게이트로 재검증한다.
>
> 형식 규약: 아래 두 표는 기계가 파싱한다(파이프 테이블, `techniques`는 `;`로 구분).
> 컬럼/헤더를 바꾸면 파서를 함께 갱신할 것.

---

## 표 1 — 실패모드 → 단계 → 기법

`stage`(파이프라인 단계)가 다른 기법은 후보에서 제외된다 — 예: `retrieval_miss`(검색 단계)에는
`CoT`(생성·추론 단계) 같은 기법을 권하지 않는다.

| failure_mode | stage | techniques |
|---|---|---|
| retrieval_miss | retrieval | top_k 상향; 청크 크기 축소(과대청크 분할); chunk_overlap 증가; 임베딩 업그레이드 후 재색인; reranker 활성화 |
| hallucination | grounding | 근거 인용 강제(citation); 컨텍스트 밖 지식 금지 프롬프트 강화; 근거 부족 시 abstain 유도; generation_model 복원 |
| over_answer | refusal | 거부 임계 강화; '근거 없으면 정보없음/Not stated' 지시 강화; no_answer few-shot 추가 |
| table_value_error | generation_reasoning | 표 셀 정확 추출 지시; 표 읽기 few-shot; 단위 명시 요구; 답 self-check 추가 |
| reasoning_error | generation_reasoning | CoT(단계적 추론); 계산 few-shot; 문제 분해 후 부분합 검증 |

단계 한글 표기: retrieval=검색 / grounding=그라운딩 / refusal=거부 / generation_reasoning=생성·추론.

---

## 표 2 — config 변경 → 흔한 회귀 (revert 1순위 후보)

candidate가 baseline에서 아래 `param`을 `bad_direction`으로 바꿨고 그 `failure_mode`가 회귀했다면,
**되돌림(revert)** 을 1순위로 제안한다(가장 직접적인 원인 후보이므로).
`direction` 의미: decrease=값↓, increase=값↑, disable=켜짐→꺼짐, change=문자열/모델 교체.

| param | bad_direction | failure_mode |
|---|---|---|
| top_k | decrease | retrieval_miss |
| chunk_size | increase | retrieval_miss |
| chunk_overlap | decrease | retrieval_miss |
| reranker_enabled | disable | retrieval_miss |
| embedding_model | change | retrieval_miss |
| generation_model | change | hallucination |
| generation_model | change | table_value_error |
| generation_model | change | reasoning_error |

---

## 표 3 — 슬라이스 집중 회귀 → 강조 기법 (보조)

특정 슬라이스에 회귀가 집중되면(해당 실패모드 케이스의 다수가 한 슬라이스) 그 슬라이스 특화 기법을 덧붙인다.

| slice | emphasis |
|---|---|
| numeric_reasoning | 수치추론 집중 → CoT·계산 few-shot 우선 |
| no_answer | 답없음 집중 → 거부 지시·no_answer few-shot 강화 |
| table_value | 표값 집중 → 표 셀 추출·단위 명시 강화 |
| body_text | 본문 집중 → 근거 인용(citation) 그라운딩 강화 |
| factoid | 사실형 집중 → 정답 span 포함 컨텍스트 확보(검색 폭) |
