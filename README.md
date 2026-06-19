# RAG Regression Gate (MCP)

_MCP 서버로 제공하는 RAG 파이프라인 회귀 진단 게이트 — 점수 하락이 아니라 어떤 실패모드가 회귀했는지 진단._

> ℹ️ **MCP 서버 제공**(`run_gate` 도구 + 룰 기반 제안 엔진) — 아래 "MCP 서버" 섹션. **범용화 완료: 2개 도메인 실증**(DART 한국 금융 100문항 + 영어 위키 SQuAD 2.0 20문항, 엔진 코드 0줄 변경). 인터페이스/경계 설계는 [`docs/portability.md`](docs/portability.md) 참조.

> **이건 "RAG 성능"을 높이는 프로젝트가 아니라, RAG 변경의 "품질 회귀"를 CI/PR에서 자동으로
> 잡는 감시 게이트 프로젝트다.** baseline의 answerable 정확도 20%는 *측정 대상*일 뿐이고, 핵심은
> **게이트가 회귀를 분별 있게 판정**하는 것 — **진짜 회귀는 FAIL로 막고, 헛개선·중립 변경은 통과**시킨다.
> 한 걸음 더: 단순 "점수 하락"이 아니라 **"어떤 실패모드가 유의하게 회귀했는지"**(retrieval_miss·hallucination 등)를 진단한다.

도메인은 **DART 전자공시(한국 금융 사업보고서)**의 표·텍스트 QA — 어렵고 오염 없는 실제 코퍼스. 하지만
DART는 **레퍼런스 인스턴스**일 뿐, 회귀 판정 엔진 자체는 도메인 무관이다(아래 "범용성" 참조).

> **작은 평가셋(100)의 노이즈를 정면으로 다룬다**: 같은 config를 **5회 반복한 노이즈 밴드 + 부트스트랩
> 신뢰구간**으로 "유의한 회귀만" 판정하고, **거짓경보 0건**(같은 config 재실행은 항상 PASS)으로 게이트
> 신뢰성을 검증했다. 작은 표본이라 못 믿는 게 아니라, 작은 표본이라서 더 정직하게 통계로 다룬다.

---

## 30초 데모 — CI 없이 로컬에서 게이트가 막는 것을 재현

게이트는 **LLM·임베딩·GPU 없이** 동작한다(`pydantic`만 필요). candidate의 채점·귀인 결과
(`run.jsonl` + `attribution.jsonl`)를 baseline과 통계 비교할 뿐이다.

```bash
pip install -r requirements-gate.txt   # pydantic 하나

# ① 중립 변경(noise r2, 무변화) → 통과
python scripts/run_gate.py --baseline examples/baseline --candidate examples/demo_neutral
#   → 🟢 GATE: PASS   (exit 0)

# ② 회귀(top_k 5→1, 검색 약화) → 차단
python scripts/run_gate.py --baseline examples/baseline --candidate examples/demo_regression
#   → 🔴 GATE: FAIL   (exit 1)
#      ❌ retrieval_miss 65→73 유의 증가 (CI [+2, +14]) → 검색 회귀
#      ❌ 정답 정확도(grounded) 0.20→0.08 유의 하락 (CI [-0.20, -0.06])
```

CI에서는 이 `exit 1`이 머지를 막는다(GitHub branch protection의 Required check).

> **코퍼스에 대해**: `data/corpus/extracted/`의 코퍼스는 DART 공시 가공본이라 저작권상 repo에
> 포함하지 않는다. **full RAG(인덱싱~생성)**를 재현하려면 [DART](https://dart.fss.or.kr)에서 해당
> 보고서 PDF를 받아 `data/corpus/raw/`에 두고 `python scripts/extract_tables.py`를 실행하면 된다.
> 단, **헤드라인인 게이트 데모(PASS/FAIL)는 코퍼스 없이 `examples/`만으로 재현된다**(위 30초 데모).

---

## MCP 서버 — Claude/Cursor에서 "두 실행 비교해줘"

같은 게이트를 **MCP 도구**로 노출한다. Claude Desktop/Cursor 같은 클라이언트가
`run_gate`를 호출해 **판정 + 실패모드 진단 + 룰 기반 제안**을 받는다.

```bash
pip install ".[mcp]"        # fastmcp는 옵션 extra (게이트 코어는 여전히 pydantic만)
python -m app.mcp.server    # stdio MCP 서버 실행
```

Claude Desktop / Cursor의 `mcpServers` 설정에 등록:

```json
{
  "mcpServers": {
    "rag-regression-gate": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/absolute/path/to/rag_regression"
    }
  }
}
```

**사용 시나리오**: Claude에게 *"이 두 RAG 실행 비교해줘"*(baseline/candidate 디렉토리 경로) →
`run_gate(baseline_dir, candidate_dir)` 호출 → **PASS/WARN/FAIL + 부트스트랩 CI + 실패모드 진단 + 제안**
(`GateResult`)을 돌려준다. 입력은 단순 경로 문자열(게이트 CLI와 동일 계약), 출력은 구조화된 Pydantic.

```
run_gate("examples/baseline", "examples/demo_regression")
 → verdict=FAIL, exit_code=1
   regressions: retrieval_miss 65→73 (CI [+2,+14]), 정답정확도 0.20→0.08
   suggestions: "[retrieval_miss] 검색 단계 회귀 … 원인 후보: top_k 5→1 → 우선 되돌림(top_k 1→5) 검토 …"
```

> **제안은 "검토 후보"지 "정답"이 아니다.** LLM이 생성하지 않고 **룰 기반 카탈로그**
> ([`docs/remediation_catalog.md`](docs/remediation_catalog.md): 실패모드→단계→기법 + config diff 역추적)로
> 결정적으로 만든다. **suggestion-only** — 게이트는 config를 자동 수정/실행하지 않으며, 모든 제안에
> "사람이 적용 후 이 게이트로 재검증" 문구가 붙는다. MCP 계층은 통계 로직을 한 줄도 재구현하지 않고
> 기존 엔진(`detect`→`gate`)을 **호출만** 한다 → CLI와 수치 동일.

### `analyze_failures` — 단일 실행 진단 (run_gate의 짝)

`run_gate`가 **두 실행을 비교**(회귀 감지)한다면, `analyze_failures`는 **한 실행을 진단**한다 —
*"바꿨더니 나빠졌나?"* 가 아니라 *"지금 어디가 약하고, 뭘 먼저 손볼까?"* 에 답한다. 운영자의 두 번째
니즈(성능을 올려야 할 때)를 위한 도구다. `run_dir` 하나만 받는다(비교 대상이 없으니 통계 검정 없음).

```
analyze_failures("examples/baseline")
 → failure_distribution: {retrieval_miss: 65, correct: 34, hallucination: 1}
   bottleneck: retrieval ("retrieval_miss가 65건으로 가장 큰 병목")
   groundedness: grounded 17 / unsupported 2 (맞았지만 근거 미실재 = 리스크)
   ragas_equivalent: context_recall 0.19, faithfulness 0.89, answer_correctness 0.20
   improvement_priorities: ① 검색(top_k↑·청크 축소) ② 표값 슬라이스 집중 … (적용 후 run_gate로 검증)
```

**RAGAS 환산 (judge 없이 결정적, 차별점)** — RAGAS의 친숙한 지표를 우리의 **결정적 측정으로 환산**한다.
LLM judge 호출이 없어 **같은 입력엔 같은 값**(재현 가능):

| RAGAS 개념 | 우리 결정적 측정 | 비고 |
|---|---|---|
| `context_recall` | `retrieval_success_strict` (gold 근거 ⊆ retrieved) | judge 없음 |
| `faithfulness` | `grounded / (grounded + unsupported)` | judge 없음 |
| `answer_correctness` | `answerable_accuracy` (grounded 기준) | judge 없음 |
| ~~`context_precision`~~ | — | **의도적 생략**: precomputed attribution(gold-free)만으론 산출 불가 |
| ~~`answer_relevancy`~~ | — | **의도적 생략**: judge 필요 → 결정성과 충돌 |

> **gold-free·결정적**: `analyze_failures`는 precomputed `attribution.jsonl`(케이스별 boolean)만 집계한다 —
> `eval_cases.jsonl`(gold)을 다시 읽지 않으므로 Phase 6의 gold 제거를 되돌리지 않는다. 새 통계/채점 로직 0.
> **suggestion-only + 닫힌 루프**: 개선 우선순위는 "검토 후보"이며, `analyze_failures`(약점 파악) →
> 개선 적용 → `run_gate`(개선 검증)로 닫는다.

---

## 메타 평가 — "분별 있게 반응한다" (Phase 5, config만 바꿔 생성)

동일 baseline에 **config 한 개씩만** 바꿔 게이트에 통과시킨 결과(임계 조작 없음):

| 변경 | answerable_acc | retrieval_miss | 게이트 | 진단 |
|---|---|---|---|---|
| (baseline) | 0.20 | 65 | — | — |
| **A. top_k 5→1** | 0.20→0.08 | 65→73 | 🔴 **FAIL** | **검색 회귀** |
| **B. reranker off→on** | 0.20→0.19 | 65→60 | 🟡 WARN | **개선 아님(헛개선)** |
| **C. overlap 150→155** | 0.20→0.19 | 65→66 | 🟡 WARN | 중립(경계) |

- **A 진짜 회귀 → 잡음.** **B "좋아질 줄 알았던" reranker → 데이터상 유의한 개선 없음(WARN), 게이트가 헛개선을 인정 안 함** — 이게 회귀 게이트의 존재 이유. **C 중립 → FAIL 회피.**
- 위 표가 핵심이며 CI·유의성은 게이트가 그대로 출력. 전체 분석: [`reports/demo_summary.md`](reports/demo_summary.md).

---

## 오해 3가지에 대한 답 (의도적 프레이밍)

**1. "정확도 20%면 RAG가 별로 아닌가?"** — 이 프로젝트의 산출물은 정확도가 아니라 **게이트의 분별력**이다.
baseline은 튜닝 안 한 측정 기준점일 뿐. 20%든 80%든, 게이트가 해야 할 일은 "변경이 이걸 **유의하게**
악화시켰는가"를 정직하게 판정하는 것이고, 위 메타 평가가 그걸 증명한다.

**2. "100문항은 표본이 작아 노이즈 아닌가?"** — 맞다. 그래서 노이즈를 **정직하게** 다룬다:
- **노이즈 밴드**: 같은 config를 5회 반복 실행해 "가만히 있어도 흔들리는 범위"를 데이터로 측정(이 repo에선 결정적이라 밴드≈0, 불안정 케이스 0).
- **부트스트랩 신뢰구간**: 케이스 단위 paired 리샘플링으로 차이의 95% CI를 구해, **CI가 0을 벗어나고(통계적 유의) 노이즈 밴드(±1 case floor)도 넘을 때만** 회귀로 판정. 둘 중 하나만이면 WARN.
- 결과: **같은 config 재실행(거짓경보 테스트)은 회귀 0건**, 합성/실제 회귀는 FAIL.

**3. "DART 전용 아닌가?"** — 회귀 엔진(노이즈밴드·부트스트랩·게이트·judge검증·실패모드 귀인)은 **도메인 무관**이고,
DART(평가셋·표추출·한국어 숫자정규화·표 도메인 taxonomy)는 **레퍼런스 인스턴스**다.
경계와 MCP 인터페이스 후보(EvalSet/RAG Adapter/Scoring Plugin/Taxonomy): [`docs/portability.md`](docs/portability.md).

---

## 차별점

- **실패모드 진단**: "점수 하락"이 아니라 `retrieval_miss`/`hallucination`/`over_answer` 중 **무엇이 유의하게 회귀했는지** 귀인. retrieval_miss는 gold 근거 ↔ 검색 청크 매칭으로 **judge 없이 결정적** 판정.
- **judge 신뢰성 검증**: 본문 채점 LLM(gpt-4o)을 gold로 검증 — 정답/오답 probe로 **judge_accuracy = 0.987**(혼동행렬 포함). 미묘 변형 probe까지 써서 거짓 고득점을 방지. → [`reports/judge_validation.json`](reports/judge_validation.json).
- **채점 전략: judge는 선택적·검증 후 사용**: 숫자/표값은 단위 정규화(조·억) + **±0.1% 허용오차로 judge 없이 결정적**, 답없음은 거부 문구 매칭으로 결정적, retrieval_miss는 gold 근거 ⊆ 검색 청크로 결정적. **judge는 본문(서술형)에만** 쓰고 그조차 gold로 검증(0.933→0.987). 회귀 게이트는 *'같은 입력엔 같은 판정'* 이 생명이라 `temperature=0`+seed로 **노이즈밴드 std=0**을 달성했고, 비결정성의 표면적을 본문으로 좁혔다.
- **RAGAS 대비**: RAGAS는 훌륭한 범용 RAG 평가 프레임워크다. 우리는 그걸 부정하는 게 아니라, *'CI 회귀 게이트'* 목적상 **결정성을 우선**했다 — *개념은 빌리되(groundedness 등) 측정은 가능한 한 결정적으로*. judge 한 번의 흔들림이 PASS/FAIL을 뒤집으면 게이트로 못 쓰기 때문. (설계 근거 전문: [`docs/JOURNEY.md` — 설계 결정](docs/JOURNEY.md))
- **groundedness 분리**: 맞은 답도 **정답값이 검색 근거에 실재(grounded)** 하는지 확인. 암기/운으로 맞은 `unsupported_correct`는 헤드라인 정확도에서 분리(모델 암기력이 RAG 점수를 부풀리지 않게).
- **no_answer 착시 방어**: answerable 정확도와 no_answer 정확도를 **항상 짝으로** 보고(전부 거부하는 시스템이 들통나도록).
- **통계적 정직 + 거짓경보 0건**: 노이즈밴드 + 부트스트랩으로 "유의한 회귀만" FAIL. **같은 config 재실행은 항상 PASS**(거짓경보 0건)로 게이트 신뢰성 검증 — 데모용 임계 조작 없음.
- **도메인 범용성 실증**: 두 도메인(DART 한국 금융 / 영어 위키 QA)에서 **같은 게이트 작동, 엔진 코드 0줄 변경**(`app/regression/*` git diff = 0). 단, 위키는 인터페이스 검증용 **미니 인스턴스**(20문항·쉬운 추출형)이고 **DART(100문항)가 메인 레퍼런스**다. → [`docs/portability.md` §5](docs/portability.md).

## 정직한 경계 (적용 범위)

과장하지 않기 위해 **못 하는 것**도 명시한다. 이 엔진은 **gold(평가셋)를 전제**로 한다 — 가진 gold에 따라 작동 범위가 갈린다:

- **정답 + 근거 라벨**(DART) → accuracy · retrieval_miss · groundedness **전 기능**(gold 근거 ⊆ retrieved, 결정적).
- **정답만**(근거 라벨 없음 — 더 흔함) → accuracy 작동, retrieval_miss는 *'정답 텍스트가 검색 청크에 있나'* 로 대체 가능(위키 `wiki_value_present`가 이 방식).
- **정답조차 없음(reference-free)** → **범위 밖**. 정답 없이 옳고 그름을 판정하려면 judge 의존이 불가피해 우리의 *결정성* 원칙과 충돌한다.

**왜 한계가 아니라 정의인가**: 회귀 감지는 본질적으로 **비교 기준**이 있어야 성립한다 — 고정 평가셋 없이 *'깨졌다'* 를 판단하는 건 원리적으로 불가능하다. 따라서 평가셋 전제는 '회귀 게이트'의 정의에 내재한 조건이다(promptfoo·RAGAS의 reference 기반 평가도 같은 전제). → 상세 [`docs/JOURNEY.md` — 설계 결정](docs/JOURNEY.md), [`docs/portability.md`](docs/portability.md).

---

## 아키텍처 — 게이트(가벼움)와 RAG 실행(무거움) 분리

```
[무거움 — CI 밖/선택]                         [가벼움 — CI 안, 활성]
 인덱싱(bge-m3 2.2GB) + 100케이스 LLM           게이트: run-log+attribution을 baseline과
 + gpt-4o judge  →  run-log + attribution  ──▶  부트스트랩 비교 → PASS/WARN/FAIL (pydantic만)
```

- **모델 1(현재)**: 무거운 RAG는 CI 밖에서 실행, candidate 산출물을 PR에 첨부. CI는 게이트만(`.github/workflows/regression-gate.yml`).
- **모델 2(전환 경로)**: `generate-candidate.yml`(Job 1, 스켈레톤)의 트리거를 `pull_request`로 바꾸고 게이트로 핸드오프하면 완전 자동 — **게이트 로직은 한 줄도 안 바뀐다.**

## 구조

```
app/rag/        수집·표추출·청킹·인덱싱·pipeline (DART/RAG)
app/evaluator/  scorer · judge · validate_judge · attribution · metrics · case_eval
app/regression/ detect(부트스트랩) · gate(PASS/WARN/FAIL)   ← 도메인 무관 엔진(2개 도메인 공유)
app/interfaces.py  플러그인 Protocol 4종(+화이트리스트)   app/adapters/  dart · wiki 구현체
app/mcp/        server(run_gate 도구, fastmcp) · suggest(룰 기반 제안 엔진)   ← MCP, 옵션 extra
scripts/        run_eval · run_attribution · measure_noise · run_scenario · run_gate · run_wiki_gate_demo
examples/       baseline / demo_neutral / demo_regression / wiki_baseline / wiki_candidate  (게이트 입력 데모)
data/wiki_eval/ SQuAD 2.0 발췌(20문항) + 코퍼스 + 라이선스(CC BY-SA 4.0)
reports/        원본 산출물(메트릭·노이즈밴드·judge검증·시나리오)   gate_runs/ 는 게이트 부산물
docs/           portability(엔진 vs 도메인 경계 + 2도메인 실증) · interfaces(플러그인 설계) · remediation_catalog(제안 근거)
```

표추출의 알려진 한계는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). (빌드 티켓·불변 규칙·기획 등 내부 작업 문서는 공개 범위에서 제외.)

**검증으로 실제 잡은 결함들**(연도누락·환각·채점 false negative·judge 무효 probe·암기정답·가짜 retrieval_miss·음수기준 증가율 등)과 그 수정 기록: [`docs/JOURNEY.md`](docs/JOURNEY.md).
