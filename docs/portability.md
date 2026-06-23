# Portability — 범용 엔진 vs DART 전용 로직

> 목적: 이 코드베이스에서 **도메인 무관 "회귀 가드 엔진"** 과 **DART/한국금융 전용** 부분을
> 분리해, 나중에 MCP 서버로 제품화할 때 무엇이 코어로 남고 무엇이 "사용자 제공
> 인터페이스"로 빠지는지 정리한다. (분석 문서 — 코드 변경 없음)

핵심 경계 한 줄:
**엔진은 "케이스별 판정(correct/실패모드) + 검색로그(retrieved_chunks)"라는 구조화된 입력만 받으면 동작한다.** 그 입력을 *어떻게 만드는지*(코퍼스·추출·RAG·채점 규칙·실패 taxonomy)가 도메인 의존부다.

---

## 1. 컴포넌트 분류표

| 컴포넌트 | 파일 | 분류 | 비고 |
|---|---|---|---|
| 회귀 판정 (paired 부트스트랩, CI) | `app/regression/detect.py` | **범용** | 메트릭/실패모드 카운트의 delta CI. 입력이 케이스별 boolean/label이면 도메인 무관 |
| 노이즈 밴드 측정 | `scripts/measure_noise.py` | **범용** | 같은 config N회 반복 → mean/std/min/max + 케이스 안정성. 도메인 무관 |
| 결합규칙 / effective_band(floor) | `detect.py` | **범용** | (CI 0미포함) AND (\|delta\|>밴드). floor의 denom(±1-case 환산)은 **런타임 모집단**(`_population_size`로 현재 run의 answerable/no_answer/all 수) — 이전엔 DART 크기(85/15/100) 하드코딩이라 타도메인 floor가 어긋났으나 도메인 무관화함(DART는 런타임값=85/15/100이라 결과 불변). 상세: [`JOURNEY.md` — 감사](JOURNEY.md) |
| 품질 게이트 PASS/WARN/FAIL + exit code | `app/regression/gate.py`, `scripts/run_gate.py` | **범용** | `FAIL_METRICS` 집합만 설정값화하면 도메인 무관 |
| judge 채점 골격 (JSON 강제 + fallback) | `app/evaluator/judge.py` | **범용(언어 의존 프롬프트)** | 메커니즘은 범용. 프롬프트 문자열만 한국어 |
| judge 검증 (gold 대비 정확도·혼동행렬·진짜오답 probe) | `app/evaluator/validate_judge.py` | **범용** | pos / neg(다른문단·미묘변형) probe 설계는 도메인 무관 |
| 메트릭 집계 + 착시방어(answerable↔no_answer 짝) | `app/evaluator/metrics.py` | **범용** | grounded/unsupported 분리, retrieval_success 이중정의 = 도메인 무관 원리 |
| groundedness 골격 (value_present) | `metrics.py: value_present` | **범용 + DART보조** | "정답값이 retrieved에 실재하는가" 원리는 범용. `_derived_source_values`는 한국 연도헤더 파싱(전용) |
| no_answer 채점 (거부=correct / over_answer) | `app/evaluator/scorer.py: score_no_answer` | **범용(언어 의존)** | 개념 범용. `_REFUSAL` 거부문구는 한국어 |
| 귀인 골격 (retrieval_miss 로그판정) | `app/evaluator/attribution.py: gold_keys/retrieved_keys/is_retrieval_miss` | **범용** | source_ref↔retrieved 메타 표/페이지 단위 매칭. 형식 규약만 맞추면 도메인 무관 |
| 케이스 스키마 (EvalCase 컨테이너) | `app/schemas.py: EvalCase, AnswerSchema` | **범용 컨테이너** | 필드 구조는 범용. `slice`/`gold_failure_type` *값*이 도메인 의존 |
| 설정 (config, 서명 해시, 시드) | `app/config.py` | **범용** | 부품 교체·인덱스 정합성 메커니즘 |
| 로컬 인덱싱/검색 인프라 | `app/rag/index.py`, `app/rag/pipeline.py` | **범용 인프라(설정 의존)** | Chroma + HuggingFace 임베딩은 config 값. bge-m3(한국어)는 *기본값*일 뿐 하드코딩 로직 아님 |
| — | | | |
| 평가셋 (100문항·슬라이스 정원·gold) | `data/eval_cases.jsonl`, `scripts/build_eval_set.py`, `scripts/check_eval_set.py`, `scripts/review_eval_set.py` | **DART 전용** | DART 보고서에서 만든 질문/정답/슬라이스(표값40·숫자20·본문25·답없음15) |
| PDF 표 추출 (마크다운화, 병합셀·다층헤더·단위·연도) | `app/rag/table_extract.py`, `scripts/extract_tables.py` | **DART 전용** | 한국 공시 PDF 구조 가정 |
| 추출 마크다운 파서 (표/연도/단위) | `app/rag/corpus_md.py` | **DART 전용** | `제N기`·`(단위: 백만원)` 등 한국 공시 포맷 파싱 |
| 표 인식 청킹 (연도헤더 반복, 섹션 키워드) | `app/rag/chunker.py` | **DART 전용(부분)** | `_SECTION_KW`(손익계산서·재무상태표…) 한국 섹션. 표-블록 청킹 자체는 일반적 |
| 숫자 정규화 (조·억·만·백만원, △·괄호 음수) | `scorer.py: _UNIT_WON, parse_signed_number, extract_amounts` | **DART 전용** | 한국 통화 단위 체계 |
| 숫자/비교 채점 규칙 (±0.1%, 이진비교) | `scorer.py: score_numeric/score_comparison` | **반(半)전용** | 비교·허용오차 *틀*은 범용, 단위/회사명 근접추출은 도메인 의존 |
| 실패 taxonomy 일부 | `schemas.py: FailureType` | **DART/표 전용 일부** | `table_value_error`·`wrong_period`·`unit_error`는 표 도메인 특화. `retrieval_miss`·`hallucination`·`correct`는 범용 |
| table_value_error vs hallucination 판정 | `attribution.py: _value_error_or_hallucination` | **DART/표 전용** | "모델값이 표의 다른 셀과 일치하는가" = 표 도메인 가정 |

---

## 2. DART 전용 → MCP "사용자 제공" 인터페이스 후보

제품화 시, 아래 전용부는 **엔진이 호출하는 플러그인 계약**으로 빠진다. 사용자는 자기 도메인 구현만
제공하고, 엔진(detect/노이즈/게이트/judge검증/메트릭 골격)은 그대로 재사용한다.

| DART 전용부 | 지금 하드코딩된 것 | MCP 인터페이스 후보 (사용자 제공) | 계약(입출력) 형태 |
|---|---|---|---|
| **평가셋** | `data/eval_cases.jsonl` + 빌드/검수 스크립트 | **EvalSet Provider** — 사용자가 `EvalCase` 스키마를 따르는 평가셋 제공 | `list[EvalCase]`. `slice`·`gold_failure_type` enum은 사용자 taxonomy로 확장 가능하게 파라미터화 |
| **PDF/표 추출 + 코퍼스** | `table_extract.py`, `corpus_md.py`, `chunker.py` | **Corpus/Ingestion Adapter** — 사용자가 자기 코퍼스를 청크(텍스트+메타)로 제공 | `chunk = {text, metadata{source_id, unit_id, is_table, ...}}`. 엔진은 추출 방식 불문 |
| **RAG 실행** | `index.py`, `pipeline.py` (Chroma+bge-m3) | **RAG Adapter** — 사용자가 `run(question) -> {retrieved_chunks(meta), answer, latency, tokens}` 제공 | 엔진은 **run-log(jsonl)** 만 소비. 검색·생성 구현은 사용자 것 (이미 구조화돼 있어 계약이 자연스러움) |
| **숫자 정규화 + 채점** | `scorer.py`의 한국 단위/거부문구/회사명추출 | **Scoring Plugin (per answer_schema)** — 사용자가 `score(answer, gold) -> {correct, detail}` 제공 | answer_schema별 매처. 통화·날짜·언어별 정규화는 플러그인 책임. 엔진은 correct boolean만 사용 |
| **거부 탐지 (no_answer)** | `_REFUSAL` 한국어 문구 | (위 Scoring Plugin에 포함) **Refusal Detector** | `is_refusal(answer) -> bool`. 언어별 |
| **실패 taxonomy + 귀인 규칙** | `FailureType` + `_value_error_or_hallucination` | **Failure Taxonomy & Attribution Plugin** | 사용자가 도메인 실패모드 + (검색됐는데 틀림)→모드 매핑 제공. **`retrieval_miss` 코어(로그 매칭)는 엔진이 제공** |
| **groundedness 보조 파서** | `_derived_source_values`(연도헤더 파싱) | (Scoring/Adapter에 포함) **Value-Presence Hook** | `value_present(case, retrieved) -> bool`. 도메인 정답값↔근거 매칭 |
| **judge 프롬프트 언어** | judge/validate 한국어 프롬프트 | **Judge Prompt/Locale** | 채점 기준·언어. judge 검증 골격(probe 설계)은 엔진 |

### 엔진 코어로 남는 것 (도메인 무관, MCP 서버 기본 제공)
- **회귀 판정**: paired 부트스트랩 CI + 노이즈밴드 결합규칙 (`detect.py`)
- **노이즈 밴드 측정**: N회 반복 통계 + 케이스 안정성 (`measure_noise.py`)
- **품질 게이트**: PASS/WARN/FAIL + exit code, 실패모드 중심 리포트 (`gate.py`)
- **judge 검증 골격**: gold 대비 정확도·혼동행렬·진짜오답 probe (`validate_judge.py`)
- **메트릭 착시방어**: answerable↔no_answer 짝, grounded/unsupported 분리, retrieval_success 이중정의 (`metrics.py`)
- **retrieval_miss 로그판정**: source_ref↔retrieved 메타 매칭 (`attribution.py`)
- **설정/재현성**: config 서명·시드 (`config.py`)

---

## 3. 경계 요약 (한 그림)

```
[사용자 제공 — 도메인 플러그인]                 [엔진 코어 — 도메인 무관]
  EvalSet Provider        ─┐
  Corpus/Ingestion Adapter ├─> run-log(jsonl) ─> 채점/귀인 골격 ─> 노이즈밴드
  RAG Adapter             ─┘    (retrieved_chunks            ─> 부트스트랩 회귀판정
  Scoring Plugin (단위·거부)     + answer + meta)            ─> 게이트 PASS/WARN/FAIL
  Failure Taxonomy/Attribution                              ─> judge 검증
  Judge Prompt/Locale                                       ─> 메트릭 착시방어
```

**제품 메시지**: "당신의 RAG(어댑터) + 평가셋(스키마) + 채점·taxonomy 플러그인만 꽂으면,
**회귀 판정·노이즈 보정·실패모드 귀인·게이트**는 그대로 쓴다." DART 구현은 그 **레퍼런스 인스턴스**.

---

## 4. 추가 분리 단서 (세분화)

표 1·2에서 "범용"으로 묶었지만, 실제 추출 시 한 겹 더 갈라지는 두 지점:

### 4.1 retrieval_miss = 범용 골격 + 도메인별 gold-근거 매칭 어댑터
`is_retrieval_miss`(집합 비교: gold 키 ∈ retrieved 키)는 **범용 골격**이다. 그러나 *gold 키를
무엇으로 보고 어떻게 뽑는지*는 도메인 의존이다. 현재 `gold_keys`는
`source_ref` 문자열을 **DART 전용 형식**(`삼성전자_2023.md#Table p82-1#…`, `…#p.420`)으로 파싱하고,
`retrieved_keys`는 `{source_file, table_id, page}` 메타에 의존한다.

→ 분리: **(범용) "gold 근거가 retrieved 집합에 있나"** + **(어댑터) gold-근거 식별자 추출/매칭**.
MCP에서는 **Gold-Evidence Matcher** 인터페이스로 빠진다:
`gold_refs(case) -> set[key]`, `retrieved_refs(run_record) -> set[key]` (키 형식은 도메인 자유,
표/페이지/문서ID 등). 엔진은 두 집합의 포함관계만 본다.

### 4.2 groundedness = 숫자형(값 매칭, 범용에 가까움) vs 서술형(judge 필요)
`value_present`는 현재 **숫자값 매칭에 특화**돼 있다(retrieved 청크의 숫자 ↔ gold 값 ±0.1%).
숫자형 정답은 이렇게 결정적으로 "근거에 실재"를 확인할 수 있어 비교적 범용적이다.
반면 **서술형(body_text) groundedness는 문자열 매칭으로 불충분**하다 — 현재는 "gold 문단(page)이
검색됐는가"라는 약한 프록시를 쓴다. 의미 수준의 "정답 내용이 근거에 실재하는가"는 **judge가 필요**하다.

→ 분리: **(범용) 숫자형 value-presence(결정적)** vs **(judge 의존) 서술형 groundedness(의미 매칭)**.
MCP에서는 **Value-Presence Hook**이 두 갈래를 가진다:
숫자/구조화 답 → 결정적 매처(도메인 단위 플러그인), 서술형 답 → judge 기반 groundedness 검증
(judge 검증 골격은 엔진, 프롬프트/기준은 사용자).

---

> 주의: 위 §1~§4는 **설계 분석**(원래 코드는 DART에 결합돼 있었다)이다. 아래 §5는 그 설계를
> 실제로 구현해 검증한 결과다.

---

## 5. 엔진 범용성 실증 (Realized)

위 설계(인터페이스 4종 + 보조 hook 2종)를 실제로 구현해 **세 도메인을 같은 엔진으로 처리**했다.

- **DART(한국 금융 사업보고서, 100문항)** — 메인 레퍼런스. **우리가 구축**한 평가셋. 병목=**검색**.
- **영어 위키 QA(SQuAD 2.0, 20문항)** — 인터페이스 검증용 **미니 인스턴스**. **우리가 발췌**. 정직히
  말해 규모가 작고(20문항) **쉬운 추출형(factoid)** 위주라, DART의 표값·숫자추론·서술형 같은
  난도/다양성은 없다. "범용성이 꽂히는지"를 보이는 용도이지 DART를 대체하지 않는다.
- **Allganize 법률·공공(한국어, 40문항)** — **외부 공개 gold**(datalama/RAG-Evaluation-Dataset-KO,
  MIT). ★ 앞의 둘과 결정적으로 다른 점: **우리가 만들지 않은 남의 gold**(질문+정답+근거문서)이고,
  병목이 **DART와 정반대(생성/그라운딩)** 다. "우리가 설계한 평가셋에만 맞춰진 엔진"이 아니라는,
  **더 강한 범용성 증거**다. 단 이것도 발췌(40) 미니 인스턴스이며 DART가 메인 레퍼런스다.

세 도메인은 같은 엔진(`app/regression/*`, `app/evaluator/metrics.py`)으로 PASS/WARN/FAIL을 낸다.
**위키·Allganize 도메인을 통째로 추가했지만 엔진은 한 줄도 안 바뀌었다** — git으로 증명:

```
$ git diff --numstat <DART-v1> HEAD -- app/regression/ app/evaluator/ \
      app/interfaces.py app/schemas.py app/config.py app/rag/ app/mcp/
(empty)        # detect / gate / 결합규칙 / noise_band / 공유코드 = 0 라인
```

도메인이 갈리는 지점은 **플러그인 구현뿐**이다(`app/interfaces.py`, `app/adapters/{dart,wiki,allganize}.py`):

| 인터페이스 | DART (금융) | Wiki (영어) | Allganize (법률·공공) |
|---|---|---|---|
| gold 출처 | 우리 구축 | 우리 발췌 | **외부 공개 gold (남이 만듦)** |
| GoldMatcher 근거 키 | `(table, file, id)` / `(page,…)` | `doc_id`(문단) | **`pid`(문서)** — 더 거침 |
| ⊆ 집합 비교 (retrieval_miss) | **동일 골격**(`gold_retrieved`, judge 없음) | **동일 골격** | **동일 골격** |
| 임베딩 (config 값) | bge-m3 | all-MiniLM-L6-v2 | bge-m3 |
| 채점·거부문구 | 한국어 | 영어 | **한국어 — DART ScoringPlugin 재활용** |
| EvalSet / 슬라이스 | 표값·숫자·본문·답없음 | factoid·no_answer | paragraph·table |
| 관측 병목 | **retrieval** | (소표본) | **grounding (DART와 정반대)** |

즉 GoldMatcher의 **근거 키만** `(table,file,id) ↔ doc_id ↔ pid`로 갈리고 **⊆ 집합 비교는 동일**하다
(gold 근거의 *단위*가 도메인마다 다르다 — DART=페이지/표, 위키=문단, Allganize=문서 — 는 것 자체가
범용성의 증거). 엔진은 `gate_fields`(케이스별 boolean)와 `noise_band`라는 동일 포맷만 받으면 도메인을
모른다. ScoringPlugin은 같은 한국어라 **Allganize가 DART 구현을 그대로 재활용**한다.

**★ 같은 진단 도구, 도메인별 다른 처방**: 동일한 `analyze_failures`(MCP)가 DART에선 병목을
*검색*으로 짚어 **top_k↑·청킹**을 권하고, Allganize에선 *그라운딩*으로 짚어 **citation 강제·
abstain 유도**를 권한다 — 도구는 하나인데 도메인의 실제 약점에 따라 처방이 갈린다.

| analyze_failures 진단 | DART (금융) | Allganize (법률·공공) |
|---|---|---|
| bottleneck_stage | **retrieval** | **grounding** |
| context_recall (≈retrieval_success) | 0.19 (낮음) | 0.975 (높음) |
| 최다 실패모드 | retrieval_miss | hallucination |
| 처방(룰 카탈로그) | top_k↑·청크 축소 | citation·abstain |

> 재현: `python scripts/run_allganize_gate_demo.py` (baseline 노이즈밴드 + top_k 5→1 candidate +
> 게이트). 결과: Allganize **게이트 FAIL** — answerable 정확도 0.55→0.30 유의 하락 + retrieval_miss
> 1→5 유의 증가를 judge 없이 검출. (위키는 `python scripts/run_wiki_gate_demo.py` → `retrieval_miss`
> 0→3 검출, 20문항 소표본이라 CI가 0에 닿아 보수적으로 WARN — 셋 다 동일한 결합규칙.)

> **Allganize의 정직한 한계**: gold 근거가 **문서 단위(pid)** 라 DART(페이지/표 단위)보다 **거칠다
> (coarser)** — 같은 문서의 다른 페이지를 검색해도 retrieval_success로 친다(groundedness도 같은
> 이유로 문서 단위). 또 일부 원문 PDF는 **전부 스캔 이미지**(추출 텍스트 0)라 해당 문서를 gold로 하는
> 문항은 무조건 retrieval_miss가 되어 **선택에서 제외**했다(이미지 미처리 범위와 일관). 40문항 **미니
> 발췌**이고 DART(100문항)가 메인 레퍼런스다. 출처/라이선스: datalama/RAG-Evaluation-Dataset-KO,
> **MIT** (`data/allganize_eval/LICENSE.md`).

### 5.1 MCP 제품화 (Realized)

§1~§4에서 "MCP 인터페이스 후보"로 적은 것 중 **게이트 코어가 실제 MCP 도구로 구현됐다**
(`app/mcp/server.py`, 의존성은 옵션 extra `[mcp]`):

- `run_gate(baseline_dir, candidate_dir) -> GateResult` — 기존 엔진(`detect_paths`→`evaluate`→
  `exit_code`)을 **호출만** 하는 얇은 래퍼. 통계 0줄 재구현, 출력 수치는 CLI와 동일.
- 제안 엔진(`app/core/suggest.py`)은 **룰 기반·결정적**: 실패모드→단계→기법 + config diff 역추적
  ([`remediation_catalog.md`](remediation_catalog.md)). LLM 미사용, suggestion-only(자동 적용 없음).

아직 "사용자 제공 플러그인"으로 추출만 한 건 도메인 어댑터(ScoringPlugin/GoldMatcher/EvalSetProvider/
RAGAdapter)다 — DART/Wiki는 코드로 꽂혀 있고, 런타임 플러그인 주입(외부 도메인 등록)은 다음 단계.

### 5.3 인터페이스 3종 (포트-어댑터) + 5중 일치 (Realized)

판정·통계 코어를 **프레임워크 중립 모듈 `app/core/`**(`service`=run_gate 코어 · `analyze`=analyze_failures
코어 · `suggest`=룰 제안)로 분리하고, 세 어댑터가 각자의 프로토콜로 그 코어를 노출한다 — **코어는 어느
프레임워크도 모른다(포트-어댑터)**:

| 인터페이스 | 어댑터 | 의존성 | 코어 |
|---|---|---|---|
| ① CLI | `scripts/run_gate.py` | pydantic만 | `app/core` |
| ② MCP | `app/mcp/server.py` | + fastmcp(`[mcp]`) | `app/core` |
| ③ REST API | `app/api/` | + fastapi/uvicorn | `app/core` |

새 인터페이스를 붙여도 코어·엔진은 0줄. 어댑터는 통계를 재구현하지 않고 같은 함수를 부른다.

**★ 5중 일치 (같은 입력 → 같은 판정)**: 게이트 수치가 5개 경로에서 동일하다 — **독립 부트스트랩**(교차검증용
재구현) = **CLI** = **in-memory**(`detect()`) = **MCP** = **REST API**. 앞 셋은 엔진 결정성(노이즈밴드 std=0)이고,
뒤 둘은 같은 코어를 부르는 어댑터다. CLI == MCP == API는 [`scripts/verify_api_equivalence.py`](../scripts/verify_api_equivalence.py)가
같은 입력(allganize baseline/candidate)에 대해 verdict·exit_code·전 메트릭·전 필드 동일을 출력해 증명한다.

**정직한 범위**: 이 API/MCP/CLI는 **run-log + attribution을 받아 판정·진단**하는 평가 서비스다. RAG 실행
자체(무거운 인덱싱·LLM·임베딩)는 범위 밖 — 사용자 RAG가 run-log를 내보내면(§5.2의 run-log 계약) 게이트가
소비한다. 과장 없이: 게이트의 본질은 *'생성'이 아니라 '소비·판정'* 이다.

### 5.2 범용성 비용과 절감 (Generic 어댑터)

3도메인을 거치며 드러난 건, 도메인마다 갈리는 게 **"키/필드 이름"뿐이고 로직은 엔진 공유**라는 점이다.
그래서 새 도메인의 비용을 "클래스 구현"에서 "설정/lambda"로 낮출 수 있다(`app/adapters/generic.py` —
순수 추가, 엔진·기존 어댑터 0줄).

**비용 (피할 수 없는 것)**: 도메인마다 4 인터페이스(EvalSet/GoldMatcher/Scoring/RAG)를 채워야 한다 —
이건 결함이 아니라 어댑터 패턴 그 자체다. 도메인의 gold 모양·근거 단위·언어가 실제로 다르기 때문.

**절감 (설정으로 낮춘 것)** — 기존 클래스를 Generic으로 재현해 **전 필드 diff 0**으로 실측:

| 인터페이스 | 도메인 전용 클래스 | Generic(설정) | 실측(Allganize) |
|---|---|---|---|
| GoldMatcher | gold/retrieved 키 추출을 클래스로 | `GenericGoldMatcher(gold_key, retrieved_key)` — **lambda 2개** | 40케이스 gold_refs/retrieved_refs 동일 |
| value_present | 도메인별 함수(거의 동일) | `make_value_present(matcher)` — **matcher 키 재사용** | 40케이스 bool 동일 |
| EvalSetProvider | `_to_case` 손으로 매핑 | `GenericEvalSetProvider(ColumnMap)` — **컬럼 매핑** | 40 EvalCase 전 필드 동일 |

즉 GoldMatcher는 lambda 2개, value_present는 matcher 재사용 한 줄, EvalSet은 컬럼 매핑으로 — **"구현"이
"설정"으로** 낮아진다. 통합 데모(`scripts/demo_generic.py`)는 Allganize를 Generic 3컴포넌트로 통째로
재구성해 attribution/metrics가 전용 어댑터와 일치함을 보이고, **`run_gate`(전용 baseline vs Generic
candidate) = PASS**로 엔진 자신이 "리팩터가 결과를 안 바꿨다"를 인증한다(기존 검증 방식 그대로).

**일반화하지 않은 것 (정직)**:
- **RAGAdapter는 일반화하지 않는다** — 도메인마다 RAG(임베딩·청킹·생성)가 본질적으로 다르다. 게다가
  실사용자는 *자기 RAG를 이미 가지고 있다* → run-log(`run.jsonl` + `attribution.jsonl`)만 내보내면
  게이트가 동작하므로 RAGAdapter 자체가 거의 불필요하다(run-log 계약으로 충분).
- **도메인 특수 로직은 전용으로 남을 수 있다** — DART의 `source_ref` tuple 파싱이나 숫자값-in-context
  정밀 매칭처럼 단순 키 추출을 넘는 로직은 Generic으로 무리하게 욱여넣지 않는다(과장 금지). Generic은
  "키/컬럼만 다른 흔한 경우"를 설정으로 낮추는 것이지, 모든 도메인 로직을 대체하는 게 아니다.

---

## 6. 정직한 경계 — gold(평가셋) 전제와 reference-free 범위 밖

§1~§5는 "엔진이 도메인을 모른다"는 범용성을 보였다. 그 범용성의 **정직한 한쪽 경계**도 함께 적는다:
이 엔진은 *'평가셋(질문+정답)을 가진 RAG'* 에 범용이며 **gold가 전제**다. 가진 gold의 수준에 따라
작동 범위가 셋으로 갈린다.

| 가진 gold | 작동 범위 | retrieval_miss / value-presence |
|---|---|---|
| **정답 + 근거 라벨** (DART=페이지/표, Allganize=문서) | accuracy · retrieval_miss · groundedness **전 기능** | gold 근거 ⊆ retrieved 집합 비교 (`is_retrieval_miss`, 결정적). 근거 *단위*는 도메인이 정함 — 문서 단위(Allganize)는 페이지/표 단위(DART)보다 **거칠다** |
| **정답만** (근거 라벨 없음 — 더 흔한 경우) | accuracy 작동, retrieval은 약한 프록시로 대체 | *'정답 텍스트가 검색 청크에 등장하나'* — 위키 `wiki_value_present`가 정답 문자열↔청크 매칭으로 이 방식(§4.2의 숫자형 value-presence와 같은 갈래) |
| **정답조차 없음** (reference-free) | **범위 밖** | 정답 없이 옳고 그름을 판정하려면 judge 의존이 불가피 → 결정성 원칙(노이즈밴드 std=0, 거짓경보 0건)과 충돌하므로 다루지 않음 |

**핵심 논리:** 회귀 감지는 본질적으로 **'기준'** 이 있어야 성립한다. 고정된 비교 기준(평가셋) 없이
*'깨졌다'* 를 판단하는 것은 원리적으로 불가능하다. 따라서 **평가셋 전제는 이 도구의 한계가 아니라
'회귀 게이트'라는 정의에 내재한 조건**이다 — 테스트셋 기반 평가 도구(promptfoo 등)와 RAGAS의
reference 기반 메트릭도 같은 전제를 공유한다. (설계 결정 맥락: [`JOURNEY.md` — 설계 결정](JOURNEY.md))
