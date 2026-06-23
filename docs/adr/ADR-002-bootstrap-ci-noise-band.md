# ADR-002: paired bootstrap CI + noise band (단순 임계값 아님)

**Status**: Accepted

## Context
평가셋이 작아(DART 100문항) 우연한 변동을 "회귀"로 오판할 위험이 크다. 단순 점수 차이(예: "−5%면 FAIL")
로는 노이즈와 진짜 회귀를 구분하지 못하고, 도메인·지표마다 노이즈 폭이 달라 거짓경보·누락이 생긴다.

## Decision
**paired bootstrap CI**(차이의 통계적 유의성)와 **noise band**(같은 config 반복 실행의 변동 범위)를
**결합**해, 둘 다 넘을 때만 FAIL로 판정한다. ±1-case floor로 미세 변동을 거른다.

## Alternatives considered
- **고정 임계값** — 도메인·지표마다 노이즈가 달라 거짓경보/누락. 기각.
- **단일 검정만** — 작은 표본에서 불안정. paired CI + band 결합이 더 보수적. 기각.

## Consequences
- (+) **거짓경보 0건**(같은 config 재실행은 항상 PASS), "유의한 회귀만" FAIL.
- (+) noise band가 std=0(결정적)일 때도 floor가 보수성을 담보.
- (−) baseline에 `noise_band.json`(반복 실행) 필요. 작은 표본의 통계적 한계는 남으며 정직히 표기한다.
- floor의 케이스 환산 denom은 도메인 무관(런타임 모집단) — [ADR-003](ADR-003-domain-adapters.md) 참조.
