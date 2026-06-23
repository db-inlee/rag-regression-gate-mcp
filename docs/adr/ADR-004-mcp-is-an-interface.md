# ADR-004: MCP는 인터페이스, 코어 아님

**Status**: Accepted

## Context
게이트를 MCP(Model Context Protocol) 도구로 노출하지만, MCP가 "핵심"인지 "하나의 접근로"인지 명확히
할 필요가 있다. (개발 중 "MCP가 굳이 필요했나"라는 의문이 직접 제기됐고, 정직히 답할 가치가 있다.)

## Decision
판정 코어는 **`app/core`**(프레임워크 무관)에 두고, MCP는 그 코어를 호출하는 **어댑터 하나**일 뿐이다.
CLI·REST API도 같은 코어를 부른다(포트-어댑터). MCP는 인터페이스 선택지이지 핵심이 아니다.

## Alternatives considered
- **MCP를 코어로(판정 로직을 MCP 서버에 결합)** — 다른 인터페이스가 MCP에 의존. 실제로 초기엔 코어 함수가
  `mcp/server.py`에 있어 REST가 fastmcp를 전이 의존했고, 리팩터로 `app/core` 분리. 기각.
- **CLI만 제공** — IDE(Cursor/Claude Code) 안에서의 자연어 워크플로를 못 살림.

## Consequences
- (+) CLI / MCP / REST가 같은 코어 공유, 각 프레임워크 의존을 격리(REST는 fastmcp 비의존), 5중 일치로 동치 증명.
- (+) 새 인터페이스를 추가해도 코어·엔진 0줄.
- (−) **정직하게**: MCP의 가치는 "범용성을 쉽게"가 아니라 **"반복 분석을 자연어로"**(run_gate/analyze_failures를
  대화에서 호출)이다. 실사용자는 자기 run-log가 있어 RAGAdapter가 거의 불필요하므로, MCP는 필수가 아닌 편의다.
