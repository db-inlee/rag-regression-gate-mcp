# ADR-001: run-log 기반 게이트 (온라인 RAG 실행 아님)

**Status**: Accepted

## Context
게이트가 RAG를 직접 실행하면 CI에서 무겁고(인덱싱·LLM·임베딩, bge-m3 2.2GB), LLM 비결정성으로
판정이 흔들리며, 특정 도메인 RAG 스택에 결합된다. 회귀 게이트는 *'같은 입력엔 같은 판정'* 이 생명이라
이 셋 모두가 치명적이다.

## Decision
게이트는 RAG를 실행하지 않고 **run-log + attribution을 소비**해 판정한다(소비 전용). 입출력 스키마는
[`artifact_contract.md`](../artifact_contract.md)로 명시한다.

## Alternatives considered
- **온라인 실행(게이트가 RAG 호출)** — CI 무겁고 느림, LLM 비결정성, 도메인 결합. 기각.
- **run-log만 (attribution 없이)** — 실패모드 귀인 불가(점수 하락만 알고 *무엇이* 회귀했는지 모름). 기각.

## Consequences
- (+) CI 경량(`requirements-gate.txt` = pydantic만), 결정적, 도메인 무관.
- (+) 같은 코어를 CLI/MCP/REST가 공유 → 5중 일치 가능([ADR-004](ADR-004-mcp-is-an-interface.md)).
- (−) 사용자가 attribution을 export해야 함 → 계약 문서로 부담을 명확화.
- 게이트의 본질은 "생성"이 아니라 **"소비·판정"**.
