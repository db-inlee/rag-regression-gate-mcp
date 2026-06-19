# Wiki 미니 도메인 평가셋 — 출처 및 라이선스

이 디렉터리의 `squad2_excerpt.jsonl` 은 **SQuAD 2.0 dev set** 에서 추출한 소량 발췌(20문항)이다.
DART(한국 금융)가 메인 레퍼런스이며, 이 발췌는 **인터페이스 범용성 실증용 미니 인스턴스**다.

## 출처 (Source)
- **데이터셋**: SQuAD 2.0 (Stanford Question Answering Dataset), dev split — `rajpurkar/squad_v2`
- **원본 파일**: <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>
- **논문**: Rajpurkar, Jia, Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL 2018.
- 본문 지문(context)은 영어 위키백과 문서에서 발췌된 것이다.

## 라이선스 (License)
SQuAD 2.0 은 **CC BY-SA 4.0** (<https://creativecommons.org/licenses/by-sa/4.0/>) 로 배포된다.
이 발췌물도 동일하게 **CC BY-SA 4.0** 를 따른다(원저작자 표시 + 동일조건 변경허락).

## 재현 (Reproduce)
`datasets` 의존성을 추가하지 않기 위해, 발췌본을 repo에 동봉한다.
원본에서 동일 발췌를 재생성하려면:

```bash
curl -L -o /tmp/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
python scripts/build_wiki_excerpt.py /tmp/dev-v2.0.json
```

선택은 결정적(데이터셋 순서, 무작위 없음): title당 적격 문단 1개씩 10개 문서에서
짧은 factoid 12개 + 적대적 no_answer 8개.

## 발췌 스키마 (squad2_excerpt.jsonl, 한 줄 = 한 문항)
| 필드 | 의미 |
|---|---|
| `qid` | SQuAD 원본 질문 id |
| `doc_id` | 위키 문서 식별자 `Title#pN` (GoldMatcher 키) |
| `title` | 위키 문서 제목 |
| `context` | 지문 문단(코퍼스에 인덱싱될 원문) |
| `question` | 질문 |
| `answers` | 정답 텍스트 목록(factoid); no_answer는 `[]` |
| `is_impossible` | `true` = 지문으로 답할 수 없음(적대적 no_answer) |
