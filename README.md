# RAG Regression Gate (MCP)

_MCP 서버로 제공하는 RAG 파이프라인 회귀 진단 게이트 — 점수 하락이 아니라 어떤 실패모드가 회귀했는지 진단._

> ℹ️ **현재 v1은 DART 레퍼런스 구현이며, MCP 서버화는 범용화 단계에서 추가 예정**(아직 미구현). 인터페이스 설계는 [`docs/portability.md`](docs/portability.md) 참조.

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
- **groundedness 분리**: 맞은 답도 **정답값이 검색 근거에 실재(grounded)** 하는지 확인. 암기/운으로 맞은 `unsupported_correct`는 헤드라인 정확도에서 분리(모델 암기력이 RAG 점수를 부풀리지 않게).
- **no_answer 착시 방어**: answerable 정확도와 no_answer 정확도를 **항상 짝으로** 보고(전부 거부하는 시스템이 들통나도록).
- **통계적 정직 + 거짓경보 0건**: 노이즈밴드 + 부트스트랩으로 "유의한 회귀만" FAIL. **같은 config 재실행은 항상 PASS**(거짓경보 0건)로 게이트 신뢰성 검증 — 데모용 임계 조작 없음.

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
app/regression/ detect(부트스트랩) · gate(PASS/WARN/FAIL)   ← 도메인 무관 엔진
scripts/        run_eval · run_attribution · measure_noise · run_scenario · run_gate · demo_table · generate_candidate
examples/       baseline / candidate / demo_neutral / demo_regression  (게이트 입력 데모)
reports/        원본 산출물(메트릭·노이즈밴드·judge검증·시나리오·실행로그)   gate_runs/ 는 게이트 부산물
docs/portability.md   범용 엔진 vs DART 전용 분리 + MCP 인터페이스 설계
```

표추출의 알려진 한계는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). (빌드 티켓·불변 규칙·기획 등 내부 작업 문서는 공개 범위에서 제외.)

**검증으로 실제 잡은 결함들**(연도누락·환각·채점 false negative·judge 무효 probe·암기정답·가짜 retrieval_miss·음수기준 증가율 등)과 그 수정 기록: [`docs/JOURNEY.md`](docs/JOURNEY.md).
