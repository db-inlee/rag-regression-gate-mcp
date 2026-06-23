# ADR-003: 도메인 어댑터 (DART 하드코딩 아님)

**Status**: Accepted

## Context
회귀 엔진이 특정 도메인(DART)에 결합되면 "범용 게이트"라 할 수 없다. 엔진(통계·판정)과 도메인 의존부
(코퍼스·추출·채점·근거 매칭·실패 taxonomy)를 분리해야 한다.

## Decision
엔진은 **케이스별 판정 + 검색로그**라는 구조화된 입력만 받고, 그 입력 생성은 **4개 도메인 플러그인**
(EvalSetProvider / GoldMatcher / ScoringPlugin / RAGAdapter, `app/interfaces.py`)이 책임진다. 엔진은
도메인을 모른다.

## Alternatives considered
- **DART 전용 로직을 엔진에 둠** — 새 도메인마다 엔진 수정 필요. 기각(범용성 상실).
- **설정 0의 완전 자동 범용** — 도메인 경계엔 어댑터가 본질적으로 필요(어댑터 패턴). 비현실적.

## Consequences
- (+) 3도메인(DART / Wiki / Allganize) **엔진 git diff = 0**으로 실증. 검색 병목(DART) vs 생성 병목
  (Allganize) 같은 정반대 프로파일도 같은 엔진이 진단.
- (+) Generic 어댑터로 "구현"을 "설정(lambda / ColumnMap)"으로 낮춤(전 필드 diff 0 재현).
- (−) 새 도메인은 어댑터 작성 비용. 단 실사용자는 자기 RAG가 이미 run-log를 냄(run-log 계약, [ADR-001](ADR-001-run-log-based-gate.md)).
- ※ 한때 noise floor의 denom이 DART 크기(85/15/100)로 하드코딩된 잔존 결합이 있었으나, **하드코딩
  감사로 스스로 발견·교정**(런타임 모집단 유도, DART 불변 증명). → [`JOURNEY.md` — 감사](../JOURNEY.md), [`portability.md`](../portability.md).
