# Architecture Decision Records (ADR)

"왜 이렇게 설계했나"를 결정·맥락·대안·결과로 남긴 기록. 이미 내린 결정의 형식화이며, 각 ADR은 반 페이지 —
핵심은 **고려한 대안과 트레이드오프**다. 상세 결함·수정 이력은 [`../JOURNEY.md`](../JOURNEY.md), 엔진 vs
도메인 경계는 [`../portability.md`](../portability.md) 참조.

| ADR | 제목 | 한 줄 요약 | Status |
|---|---|---|---|
| [001](ADR-001-run-log-based-gate.md) | run-log 기반 게이트 | RAG를 실행하지 않고 run-log+attribution을 **소비**해 판정(경량·결정적·도메인 무관) | Accepted |
| [002](ADR-002-bootstrap-ci-noise-band.md) | bootstrap CI + noise band | 단순 임계값 대신 **유의성 AND 노이즈밴드** 결합 → 거짓경보 0건, 유의한 회귀만 FAIL | Accepted |
| [003](ADR-003-domain-adapters.md) | 도메인 어댑터 | 엔진은 도메인을 모르고 4개 플러그인이 도메인 의존부 담당 → 3도메인 엔진 0줄 실증 | Accepted |
| [004](ADR-004-mcp-is-an-interface.md) | MCP는 인터페이스 | 코어는 `app/core`(프레임워크 무관), MCP/CLI/REST는 어댑터 — MCP는 핵심이 아닌 편의 | Accepted |
