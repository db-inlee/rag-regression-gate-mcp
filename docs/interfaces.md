# Interfaces — 범용화 1단계 설계 (시그니처만, 구현 없음)

> [`portability.md`](portability.md)의 "엔진 범용 / DART 레퍼런스" 경계를 **플러그인 인터페이스**로
> 구체화한다. 이 문서는 **설계(시그니처)** 만 정의한다 — 구현 로직은 아직 없다.

## ★ 핵심 원칙 — 갈아엎지 않고 감싼다 (wrap, not rewrite)

- 각 인터페이스의 DART 구현체는 **기존 함수를 호출만** 한다. `scorer.score_case`, `pipeline.Pipeline.run`,
  `attribution.gold_keys/is_retrieval_miss` 등의 **시그니처·동작은 불변**.
- 추상화 도입 후 **DART baseline을 wrapper로 재생성 → 기존 run-log/attribution과 동일해야 한다.**
  동일성 검증은 **우리 게이트로** 한다(같은 config → 회귀 0건이면 리팩터가 행동을 안 바꿨다는 증거).
  즉 **"회귀 가드가 자기 리팩터링을 회귀 검증"** 한다.

---

## 0. 공유 타입

### 재사용 (`app/schemas.py`, 변경 없음)
- `EvalCase` — 케이스 컨테이너. 필드 그대로(`id, question, contexts, answer_schema, expected_answer,
  answer_type, slice, gold_failure_type, source_ref, …`).
- `ExpectedAnswer` = `NumericAnswer | ComparisonAnswer | TextAnswer | NoAnswer` — 도메인이 변형 추가 가능.
- `FailureType`, `Slice`, `AnswerSchema` — 범용화 시 **고정 `Literal` → 도메인 config 주입**으로 파라미터화
  (값만 도메인 의존, 구조는 범용).

### 신규 — 기존 dict 형태를 타입으로 명문화 (구현 변경 아님, 계약 명시)
```python
class ScoreResult(TypedDict):
    correct: bool | None        # None = judge로 위임(서술형). 현 score_case 반환과 동일
    score_detail: dict          # over_answer / comparison_parse / refused 등 메타

class RetrievedChunk(TypedDict):
    text: str
    metadata: dict              # 도메인 자유: {source_file, table_id, page, ...} / {doc_id, section, ...}

class RunLogEntry(TypedDict):
    id: str
    slice: str
    question: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None
    latency_ms: float
    llm_calls: int
    token_usage: dict

GoldRef = Hashable              # 근거 식별자 (현 DART: tuple ("table", file, table_id))
```

---

## 1. ScoringPlugin

```python
class ScoringPlugin(Protocol):
    def score(self, answer: str | None, gold: ExpectedAnswer, case: EvalCase) -> ScoreResult: ...
    def is_refusal(self, answer: str) -> bool: ...   # no_answer / "정보 없음" 판정 (언어별)
```

- **I/O 타입**: 입력 `answer`(run-log의 모델 답), `gold`(=`case.expected_answer`), `case`(EvalCase) →
  `ScoreResult`. **schemas 재사용**(`ExpectedAnswer`/`EvalCase`); `ScoreResult`는 현 `score_case` 반환
  형태(`{correct, score_detail}`) 그대로.
- **DART 구현 매핑**: `DartScoringPlugin.score()` → 내부에서 **`app/evaluator/scorer.py:score_case(case, answer)`**
  호출. 숫자 정규화(`_UNIT_WON`·`extract_amounts`·`parse_signed_number`), table_value ±0.1%,
  comparison 이진, no_answer 거부(`is_refusal`←`_REFUSAL`)를 **그대로 위임**.
- **2번째 도메인(영어 위키 QA)**: 통화(조·억) 정규화 불필요 → 숫자/날짜/엔티티 정규화 플러그인으로 교체,
  `is_refusal`은 영어 문구("not stated in the document"). 서술형은 그대로 judge 위임.
- **엔진이 도는 법**: `detect`/`metrics`는 `ScoreResult.correct`(boolean)만 소비. 단위·언어를 전혀 모름 →
  플러그인이 정규화를 책임지므로 엔진은 도메인 무관.

## 2. GoldMatcher

```python
class GoldMatcher(Protocol):
    def gold_refs(self, case: EvalCase) -> set[GoldRef]: ...          # 근거 식별자 추출
    def retrieved_refs(self, entry: RunLogEntry) -> set[GoldRef]: ...  # 청크 메타 → 식별자

# 엔진(범용) 합성 — 도메인 코드 아님:
#   is_retrieved(case, entry)  := gold_refs(case) ⊆ retrieved_refs(entry)   (전부 포함)
#   is_retrieval_miss          := gold_refs ≠ ∅  and  not is_retrieved      (gold 없으면 N/A)
```

- **I/O 타입**: `GoldRef`는 도메인이 정의하는 해시 가능 키(현 DART: `tuple`). 엔진은 **두 집합의 포함관계만**
  본다(`is_retrieved(gold_ref, retrieved_chunks)->bool`의 일반화 = 집합 ⊆).
- **DART 구현 매핑**: `gold_refs` ← **`attribution.py:gold_keys(case)`** (`source_ref` 파싱:
  `삼성전자_2023.md#Table p82-1#…` → `("table", file, "p82-1")`). `retrieved_refs` ←
  **`attribution.py:retrieved_keys(entry)`** (metadata `{source_file, table_id, page}`).
  엔진 `is_retrieval_miss`는 **`attribution.py:is_retrieval_miss`** 그대로(표 단위, comparison은 둘 다 필요).
- **⊆ 가 기존 comparison 동작과 정확히 동일**: comparison 케이스는 `gold_refs`에 **표 2개**가 들어간다.
  `gold_refs ⊆ retrieved_refs` = "**둘 다 포함**" → **하나라도 누락이면 ⊄ → retrieval_miss**. 이는 현
  `is_retrieval_miss`의 `not all(g in have for g in golds)`(=하나라도 빠지면 miss)와 **수식적으로 일치**한다.
  → **§6 검증에서 comparison 케이스 8개의 attribution이 diff 0인지 반드시 확인**(단일/표/비교 모두 동일성 보장).
- **2번째 도메인**: `GoldRef = (doc_id,)` 또는 `(doc_id, section)`; `retrieved_refs`는 `chunk.metadata["doc_id"]`.
  **source_ref 형식만 도메인**, 포함관계 판정(엔진)은 한 줄도 안 바뀜.
- **엔진이 도는 법**: `retrieval_miss` 귀인은 **judge 없이** 집합 비교만으로 결정적. DART든 위키든 동일.

## 3. EvalSetProvider

```python
class EvalSetProvider(Protocol):
    def load(self) -> list[EvalCase]: ...
```

- **I/O 타입**: → `list[EvalCase]` (schemas 그대로). `slice`/`gold_failure_type` enum은 **도메인 taxonomy로
  파라미터화**(고정 Literal → 주입식).
- **DART 구현 매핑**: `DartEvalSetProvider.load()` → **`data/eval_cases.jsonl`** 읽어 `EvalCase.model_validate`.
  (생성·검수는 `build_eval_set.py`/`check_eval_set.py`/`review_eval_set.py`가 만든 **고정셋**; 런타임 로딩만 감쌈.)
- **2번째 도메인**: `load()`가 위키 QA 셋(jsonl/HF dataset)을 `EvalCase`로 매핑. `slice ∈ {factoid, multi_hop,
  no_answer, …}`, taxonomy도 도메인. **평가셋은 고정**(회귀는 baseline 대비)이라는 불변 규칙은 유지.
- **엔진이 도는 법**: 엔진은 `case.id/slice/answer_type/expected_answer`만 본다. 평가셋 출처·생성법 불문.

## 4. RAGAdapter

```python
class RAGAdapter(Protocol):
    def run(self, question: str) -> RunLogEntry: ...
```

- **I/O 타입**: `question: str` → `RunLogEntry`. 현 `Pipeline.run` 반환(`retrieved_chunks/answer/latency_ms/
  llm_calls/token_usage`)과 거의 일치 → 얇은 필드 매핑(+`id/slice`는 호출부가 케이스에서 채움).
- **DART 구현 매핑**: `DartRAGAdapter.run()` → **`app/rag/pipeline.py:Pipeline(config).run(question)`** 호출 후
  `RunLogEntry`로 매핑(Chroma + bge-m3 검색 + gpt-4o-mini 생성). `index.py`/`pipeline.py` **불변**.
- **2번째 도메인**: `run()`이 위키 인덱스(다른 임베딩/벡터스토어) + 다른 LLM으로 검색·생성, `metadata` 키만
  도메인에 맞게(`doc_id` 등). 엔진은 `retrieved_chunks` **구조**만 본다.
- **엔진이 도는 법**: run-log가 **모든 하위(채점·귀인·게이트) 입력**. 엔진은 로그 출처를 모름(모델 1/2 공통).

---

## 5. 엔진이 인터페이스 위에서 도는 전체 흐름

```
EvalSetProvider.load() → [EvalCase…]
   └ 각 case:
        RAGAdapter.run(case.question) ──────────────► RunLogEntry (retrieved_chunks + answer + ops)
        ScoringPlugin.score(answer, gold, case) ────► ScoreResult.correct
        GoldMatcher: gold_refs ⊆ retrieved_refs ────► retrieval_miss 여부 (judge 無)
        ValuePresence hook(숫자=결정/서술=judge) ────► grounded 여부
   └ per-case 판정 = enriched attribution  (현 gate_fields 형태:
        {id, slice, answerable, correct, value_present, retrieval_strict_ok, over_answer, primary_failure})
                                   │
                                   ▼   ← 여기부터 100% 도메인 무관 엔진
        detect(paired 부트스트랩 CI) · noise_band(N회) · metrics(착시방어) · gate(PASS/WARN/FAIL)
```

- **위 점선 위쪽(플러그인 4종 + 2개 보조 훅)** 만 도메인이 제공. **아래쪽 엔진**은 그대로.
- 보조 훅 2개(이번 4종에 포함 안 됨, 다음 설계 — portability §4 참조):
  **Value-Presence Hook**(`value_present`, 숫자 결정형/서술 judge형), **Failure Taxonomy & Attribution Plugin**
  (`table_value_error` 등 도메인 모드; `retrieval_miss`/`hallucination`/`correct`는 엔진 제공).

> **⚠️ Value-Presence는 4개 인터페이스보다 도메인 의존이 복잡 — 2번째 도메인의 잠재 병목.**
> ScoringPlugin·GoldMatcher·EvalSetProvider·RAGAdapter 4종은 **순수 위임**으로 깔끔히 분리된다.
> 그러나 Value-Presence(groundedness)는 갈래가 둘이다:
> - **DART(숫자형) = 결정형** — "숫자값이 retrieved에 ±0.1% 실재"를 단위 정규화로 결정적 판정(현 `value_present`).
> - **서술형 도메인(영어 위키 QA 등) = judge형** — "정답 *의미*가 근거에 있나"는 문자열 매칭으로 불충분,
>   **judge 호출 + 기준 설계**가 필요(현재 본문은 "gold 문단 검색됨"이라는 약한 프록시만 사용).
>
> 따라서 서술 중심 2번째 도메인을 꽂을 때, 4개 인터페이스는 그대로지만 **Value-Presence는 judge 기반
> groundedness를 새로 설계**해야 한다. 이 훅을 4종과 분리해 별도 단계로 다루는 이유.

---

## 6. 마이그레이션 & 회귀 검증 (행동 불변 보장)

1. **Protocol 정의** (위 시그니처). 엔진이 구체 클래스 대신 Protocol에 의존(주입).
2. **DART wrapper 작성** — 로직 추가 없이 기존 함수 **위임만**:
   `DartScoringPlugin→score_case`, `DartGoldMatcher→gold_keys/retrieved_keys/is_retrieval_miss`,
   `DartEvalSetProvider→eval_cases.jsonl`, `DartRAGAdapter→Pipeline.run`.
3. **DART wrapper로 baseline 재생성** → 기존 `examples/baseline`·attribution과 **diff 0** 기대.
4. **게이트로 검증**: `run_gate.py --baseline examples/baseline --candidate <wrapper로 만든 동일 config 실행>`
   → **PASS / 회귀 0건**이어야 함. 차이가 나오면 wrapper가 행동을 바꾼 것 → 추상화 결함.
5. 동일성 확인 후에만 2번째 도메인 플러그인 추가.

> 불변식: wrapper는 **순수 위임**(파라미터 변환만). 기존 DART 산출(run-log·attribution·메트릭)이
> 비트 단위로 같아야 한다 — 우리 회귀 게이트가 그 보증 도구다.

## 7. 2번째 도메인(영어 위키 QA) 체크리스트 — "무엇만 바꾸나"

| 바꾸는 것 (도메인 플러그인) | 안 바꾸는 것 (엔진 그대로) |
|---|---|
| `EvalSetProvider` — 위키 QA → EvalCase, 도메인 slice/taxonomy | `detect`(부트스트랩) · `gate` · `metrics` · `noise_band` |
| `RAGAdapter` — 위키 인덱스·임베딩·LLM, metadata `doc_id` | retrieval_miss **집합 포함 로직** |
| `ScoringPlugin` — 영어 정규화 + 영어 거부문구 | judge **검증 골격**(probe 설계) |
| `GoldMatcher` — gold_ref=doc/section id 매칭 | config 서명·시드·재현성, 게이트 exit code |

→ **도메인이 꽂는 건 4개 인터페이스(+2 훅), 엔진은 0줄 변경.** DART는 이 인터페이스들의 **레퍼런스 구현**.

---

> 다음 단계(이 문서 승인 후): Protocol 코드 골격 + DART wrapper 작성 → §6 게이트 검증으로 행동 불변 확인.
> (이 문서는 설계만 — 아직 코드 없음.)
