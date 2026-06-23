# Artifact Contract — 게이트 입출력 스키마

게이트가 **소비(consume)** 하는 입력과 **생성(produce)** 하는 출력의 필드·타입·의미 계약.
모든 예시는 현재 코드로 재생성된 `examples/`·`gate_runs/` 실측값이다(추측 없음).

---

## 1. 개요

게이트는 **RAG를 실행하지 않는다** — 이미 생성된 run-log와 attribution을 소비해 판정한다.

```
[ baseline/  run.jsonl + attribution.jsonl + noise_band.json ]
[ candidate/ run.jsonl + attribution.jsonl                   ]
                         │
                         ▼   detect_paths → evaluate  (부트스트랩 CI + 노이즈밴드)
                         │
              [ gate_result : PASS | WARN | FAIL (+exit_code) ]
```

판정에 필요한 건 사실상 `attribution.jsonl`(+baseline `noise_band.json`)뿐이다 — gold/eval-set도,
run.jsonl조차도 필요 없다(§5의 `test_detect_paths_gold_free`가 이를 강제). run.jsonl은
`analyze_failures` 진단·디버깅·재현용이다.

---

## 2. 입력 계약 (consume)

### 2.1 `run.jsonl` — JSONL: header 1줄 + case N줄

**header** (`type="header"`): `run_id`, `domain`, `embedding_model`, `generation_model`,
`n_cases`, `config{…}`(RagConfig fingerprint — 전체 스펙은 `app/config.py`).

**case** (`type="case"`): `id`, `slice`, `question`, `retrieved_chunks[]`, `answer`,
`latency_ms`, `llm_calls`, `token_usage{}`.
- `retrieved_chunks[]`: `{text, metadata{…}, distance}` — `metadata`는 도메인별이며
  **GoldMatcher가 읽는 키**(예: allganize는 `pid`, wiki는 `doc_id`)를 포함해야 한다.

실측 1줄씩 (allganize, 축약):
```json
{"type":"header","run_id":"allganize_noise_1a13b95a9b06_r1","domain":"allganize","embedding_model":"BAAI/bge-m3","generation_model":"gpt-4o-mini","n_cases":40,"config":{"chunk_size":1000,"chunk_overlap":150,"top_k":5,"embedding_model":"BAAI/bge-m3","generation_model":"gpt-4o-mini","judge_model":"gpt-4o","reranker_enabled":false}}
{"type":"case","id":"allganize_001","slice":"paragraph","question":"A가 …설명하였나요?","retrieved_chunks":[{"text":"2) 피고 A의 I 업데이트 …","metadata":{"domain":"law","file_name":"[민사] …pdf","page":4,"pid":"p_42","doc_id":"p_42"},"distance":0.5556}],"answer":"…2017년 1월 23일에 배포…","latency_ms":3396.9,"llm_calls":1,"token_usage":{"prompt_tokens":2884,"completion_tokens":88,"total_tokens":2972}}
```

### 2.2 `attribution.jsonl` — JSONL: case당 1줄 ★ 게이트 핵심 입력

필수 8필드 (§5의 `test_attribution_fields_present`가 다도메인에서 강제):

| 필드 | 타입 | 의미 |
|---|---|---|
| `id` | str | 케이스 식별자 |
| `slice` | str | 평가 슬라이스 (paragraph / table / …) |
| `primary_failure` | str | 실패모드 귀인: `correct` \| `retrieval_miss` \| `hallucination` \| `over_answer` |
| `answerable` | bool | 답할 수 있는 질문인가 |
| `correct` | bool | 정답인가 |
| `value_present` | bool | 정답 근거가 검색됐나 (groundedness) |
| `retrieval_strict_ok` | bool | gold 근거 ⊆ 검색 청크인가 |
| `over_answer` | bool | 답없음인데 답했나 |
| `attribution_detail` | dict | 부가 정보(옵션; retrieval_miss 시 gold/retrieved 키 등) |

실측 2줄 (correct / retrieval_miss):
```json
{"id":"allganize_001","slice":"paragraph","primary_failure":"correct","attribution_detail":{},"answerable":true,"correct":true,"value_present":true,"retrieval_strict_ok":true,"over_answer":false}
{"id":"allganize_036","slice":"table","primary_failure":"retrieval_miss","attribution_detail":{"gold_keys":["p_14"],"retrieved":["p_10","p_15"]},"answerable":true,"correct":false,"value_present":false,"retrieval_strict_ok":false,"over_answer":false}
```
→ 사용자는 자기 RAG 결과를 **이 8필드로 매핑**하면 게이트에 물릴 수 있다.

### 2.3 `noise_band.json` — baseline에만 필요

`n_runs`, `domain`, `config{}`, `run_ids[]`, `metric_band{ metric: {mean, std, min, max, n} }`.
같은 config를 반복 실행한 메트릭 분포 → floor/band 계산 기준. `std=0`이면 결정적.

실측(축약):
```json
{"n_runs":3,"domain":"allganize","run_ids":["…_r1","…_r2","…_r3"],
 "metric_band":{"answerable_accuracy":{"mean":0.525,"std":0.0204,"min":0.5,"max":0.55,"n":3}, "...":{}}}
```

---

## 3. 출력 계약 (produce)

### 3.1 `gate_result.json` (`gate_runs/gate_<candidate>.json`)

최상위: `gate`("PASS"|"WARN"|"FAIL"), `baseline_run`, `candidate_run`, `results[]`, `gate_detail{}`.

`results[]` 각 행: `metric`, `kind`(`prop`|`count`), `regression_dir`(`down`|`up`), `baseline`,
`candidate`, `delta`, `delta_cases`, `ci_low`, `ci_high`, `effective_band_cases`,
`significant`(bool), `direction`(`regression`|`improvement`|`warn`|`no_change`).

실측 1행 (allganize candidate, 현재 코드 = denom 런타임):
```json
{"metric":"answerable_accuracy","kind":"prop","regression_dir":"down","baseline":0.55,"candidate":0.3,
 "delta":-0.25,"delta_cases":10.0,"ci_low":-0.4,"ci_high":-0.125,"effective_band_cases":1.0,
 "significant":true,"direction":"regression"}
```
(`delta_cases = |−0.25|×40 = 10.0`, `effective_band_cases = max(std 0.0204×40, 1.0) = 1.0` — 모집단 40 기준.)

**결합 규칙(한 줄)**: 회귀 = `significant`(부트스트랩 CI가 0 제외) **AND** `beyond_band`(`delta_cases > effective_band_cases`). 한 조건만이면 WARN.

**exit_code 계약**: `FAIL → 1`, `WARN/PASS → 0` (CI 머지 차단용).
**CI에서 이렇게 보입니다**: FAIL 시 PR에 게시되는 사람용 리포트 예시 → [`examples/pr_comment.md`](../examples/pr_comment.md).

**★ 도메인 무관 floor**: `delta_cases`·`effective_band_cases`의 케이스 환산은 **현재 run의 실제
모집단**을 쓴다 — Wiki `×12` / Allganize `×40` / DART `×85`(answerable 기준). 이전엔 DART 크기로
하드코딩돼 타도메인 floor가 어긋났으나 도메인 무관화함(상세: [`portability.md`](portability.md), [`JOURNEY.md` — 감사](JOURNEY.md)).

### 3.2 `metrics.json` (단일 run 헤드라인)

`run_id`, `answerable_accuracy`, `answerable_total`, `grounded_correct`, `unsupported_correct`,
`no_answer_accuracy`, `over_answer_rate`, `retrieval_success_strict`, `retrieval_miss_count`,
`failure_distribution{}`.

실측 (allganize baseline):
```json
{"run_id":"allganize_noise_1a13b95a9b06_r1","answerable_accuracy":0.55,"answerable_total":40,
 "grounded_correct":22,"unsupported_correct":0,"no_answer_accuracy":null,"over_answer_rate":null,
 "retrieval_success_strict":0.975,"retrieval_miss_count":1,
 "failure_distribution":{"correct":22,"hallucination":17,"retrieval_miss":1}}
```

---

## 4. 사용자 가이드 (export 방법)

내 RAG를 게이트에 물리려면:
1. run마다 **`attribution.jsonl`** 을 §2.2의 8필드로 생성(case별 채점·귀인 결과).
2. baseline / candidate 두 디렉토리로 둔다. baseline은 **`noise_band.json`** 도(같은 config 반복 실행).
3. `python scripts/run_gate.py --baseline <baseline_dir> --candidate <candidate_dir>`
   (또는 MCP `run_gate` / REST `POST /evaluate` — 같은 코어, 같은 수치).

**★ 정직한 경계**: `attribution.jsonl`을 만들려면 case별 **채점·귀인**(정답 판정·실패모드·근거 매칭)이
필요하고, 이는 도메인 의존(ScoringPlugin / GoldMatcher의 역할)이다 — 게이트가 대신 하지 않는다.
게이트는 "이미 귀인된 결과"를 소비할 뿐이다. (RagConfig 전체 스펙은 `app/config.py` 참조.)

---

## 5. 계약과 테스트의 연결

이 계약은 문서일 뿐 아니라 **테스트로 강제**된다 (`tests/test_attribution_contract.py`):
- `test_attribution_fields_present` — 모든 attribution 레코드가 필수 8필드를 올바른 타입으로 보유(allganize·DART 다도메인 파라미터화).
- `test_detect_paths_gold_free` — `attribution.jsonl`(+noise_band)만으로 판정 — run.jsonl·gold·eval_cases 없이 동작.

→ **계약 위반 시 CI가 잡는다.**
