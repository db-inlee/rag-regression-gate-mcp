"""One-off surgical patch of data/eval_draft.jsonl (post-review).

Discards 7 flagged cases and regenerates 7 replacements in place (same ids, so
the 100-line count / slice distribution stay stable):

  body_text          : case_082, case_087   (없음-answer / non-fact)
  numeric (growth)   : case_052             (negative base year -> meaningless %)
  no_answer          : case_072..075        (unverified "missing item" traps)

Replacement rules (enforced here):
  * growth: base-year value must be > 0.
  * no_answer "없는 항목": the item is grep-verified absent in the target report.
  * body_text: answer must appear verbatim in the gold paragraph; no number-only
    answers, no "없음", no subsidiary-subject paragraphs; fresh page (not reused).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.build_eval_set as B
from app.env import require_openai_key
from app.rag.corpus_md import load_paragraphs, load_tables, parse_number, year_columns
from app.schemas import EvalCase

DRAFT = Path("data/eval_draft.jsonl")
EXTRACTED = Path("data/corpus/extracted")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------- #
# Replacement builders
# --------------------------------------------------------------------------- #

def growth_replacement(file: str, year: int) -> dict:
    """연결 매출액 YoY growth for the report year (base year must be positive)."""
    tables = load_tables(EXTRACTED / file)
    for tb in tables:
        if B.statement_type(tb.title) != "income":
            continue
        yc = year_columns(tb.headers)
        if year not in yc or (year - 1) not in yc:
            continue
        for row in tb.rows:
            if B._match_item(row[0], B.INCOME_ITEMS) != "매출액":
                continue
            cur, prev = parse_number(row[yc[year]]), parse_number(row[yc[year - 1]])
            if cur is None or prev is None or prev <= 0:
                continue
            growth = round((cur - prev) / prev * 100, 2)
            case = B._growth_case(tb, row, "매출액", year, growth)
            case["needs_review"] = True
            return case
    raise RuntimeError(f"no positive-base 매출액 growth found in {file}")


def no_answer_replacement(file: str, year: int, company: str, item: str, core_item: str) -> dict:
    text = (EXTRACTED / file).read_text(encoding="utf-8")
    hits = text.count(core_item)
    assert hits == 0, f"'{core_item}' appears {hits}x in {file}; not a safe absent-item trap"
    return {
        "company": company,
        "source_doc": file,
        "fiscal_year": year,
        "question": f"{company} {year}년 보고서 기준, 보고된 {item}은 얼마인가?",
        "contexts": [],
        "answer_schema": "no_answer",
        "expected_answer": {"sentinel": "정보 없음"},
        "answer_type": "unanswerable",
        "slice": "no_answer",
        "gold_failure_type": "correct",
        "source_ref": file,
        "needs_review": True,
    }


def _normalize(s: str) -> str:
    return " ".join(s.split())


# Boilerplate / table-of-contents / cover-page markers — paragraphs or answers
# containing these are not substantive facts.
_BOILER = ("작성기준일 이후", "기타사항", "해당사항 없", "..........", "목 차",
           "금융위원회", "귀중", "다음과 같습니다", "아래와 같습니다", "참조하")
# Prose business sections to prefer over registry/share-history list dumps.
_PROSE_KW = B.BODY_SECTION_KEYWORDS + ("기술", "제품", "생산", "공급", "시장",
                                       "고객", "서비스", "설립", "연구", "전략")
_AGG_WORDS = ("가장", "최대", "최소", "합계", "평균", "몇 ", "총 ", "각각")


def _is_listdump(text: str) -> bool:
    return (text.count("※") >= 2
            or len(re.findall(r"\d{2,4}[.\-/]\d{1,2}", text)) >= 3
            or "발행일자" in text or "주당발행가액" in text)


def _is_numberish(s: str) -> bool:
    # whole answer is just a figure/date/quantity (belongs in table_value, not body_text)
    return bool(re.fullmatch(r"[\d,.\-'/%()년월일억조만천백십대건명개월주원톤㎡㎞㎏ ]+", s.strip()))


def _gen_body_text(client, file: str, company: str, year: int, used_pages: set[str]) -> dict:
    """Pick a fresh, substantive prose paragraph and generate a verbatim-grounded Q/A."""
    paras = [p for p in load_paragraphs(EXTRACTED / file)
             if str(p.page) not in used_pages
             and "종속기업" not in p.text and "자회사" not in p.text
             and not any(b in p.text for b in _BOILER)
             and not _is_listdump(p.text)
             and 200 <= len(p.text) <= 1500]
    # prefer prose business sections, then page order
    paras.sort(key=lambda p: (0 if any(k in p.text for k in _PROSE_KW) else 1, p.page))

    for p in paras:
        core, key_points = _llm_verbatim(client, company, year, p.text[:1600])
        if not core:
            continue
        norm = _normalize(p.text)
        if any(w in core for w in _AGG_WORDS):
            continue  # aggregation/superlative question needs reasoning, not retrieval
        if any("없" in k for k in key_points):
            continue  # no "없음" answers
        if any(b in k for k in key_points for b in _BOILER):
            continue  # boilerplate / section-title answer, not a fact
        if not all(_normalize(k) in norm for k in key_points):
            continue  # must be literally present
        if all(_is_numberish(k) for k in key_points):
            continue  # not a number/date-only (table-style) answer
        return {
            "company": company,
            "source_doc": file,
            "fiscal_year": year,
            "question": f"{company} {year}년 보고서 기준, {core}",
            "contexts": [p.text],
            "answer_schema": "text",
            "expected_answer": {"key_points": key_points},
            "answer_type": "answerable",
            "slice": "body_text",
            "gold_failure_type": "correct",
            "source_ref": f"{file}#p.{p.page}",
            "needs_review": True,
        }
    raise RuntimeError(f"no verbatim-groundable paragraph found in {file}")


def _llm_verbatim(client, company: str, year: int, paragraph: str) -> tuple[str | None, list[str]]:
    prompt = (
        f"다음은 {company}의 {year}년 사업보고서 본문 한 문단이다. 이 문단만 근거로 "
        "답이 하나로 정해지는 사실 질문 1개와, 그 답을 만들어라.\n"
        "규칙(반드시):\n"
        "- key_points의 각 항목은 문단에 등장하는 표현을 **그대로(verbatim) 복사**하라(요약·재서술 금지).\n"
        "- 질문이 단일 사실이면 key_points는 1개. 최대 2개.\n"
        "- 질문에는 회사명·연도·'당사'를 넣지 마라(시스템이 접두사로 붙인다).\n"
        "- 정답이 '없다/해당없음'이 되는 질문, 숫자만으로 답하는 질문은 만들지 마라.\n"
        '- JSON으로만: {"question": "...", "key_points": ["..."]}\n\n문단:\n' + paragraph
    )
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=B.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content)
            q = str(data["question"]).strip()
            kp = [str(x).strip() for x in data["key_points"] if str(x).strip()]
            if q and kp:
                return q, kp[:2]
        except Exception:  # noqa: BLE001
            pass
    return None, []


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    rows = [json.loads(l) for l in DRAFT.read_text(encoding="utf-8").splitlines() if l.strip()]
    byid = {c["id"]: c for c in rows}

    # used body_text pages per file (to pick fresh ones)
    used: dict[str, set[str]] = {}
    for c in rows:
        if c["slice"] == "body_text":
            used.setdefault(c["source_doc"], set()).add(c["source_ref"].split("#p.")[1])

    require_openai_key()
    from openai import OpenAI
    client = OpenAI()

    replacements: dict[str, dict] = {}

    # growth (positive base year): case_052 had 하이닉스_2024 매출총이익 (negative
    # base); case_044 had 하이닉스_2024 영업이익 (negative base) — same defect.
    replacements["case_052"] = growth_replacement("하이닉스_2024.md", 2024)  # 매출액
    replacements["case_044"] = growth_replacement("하이닉스_2025.md", 2025)  # 매출액

    # no_answer (grep-verified absent items; same target reports as discarded)
    replacements["case_072"] = no_answer_replacement(
        "삼성전자_2024.md", 2024, "삼성전자", "직원 1인당 영업이익", "1인당 영업이익")
    replacements["case_073"] = no_answer_replacement(
        "삼성전자_2025.md", 2025, "삼성전자", "직원 1인당 당기순이익", "1인당 당기순이익")
    replacements["case_074"] = no_answer_replacement(
        "하이닉스_2023.md", 2023, "하이닉스", "직원 1인당 영업이익", "1인당 영업이익")
    replacements["case_075"] = no_answer_replacement(
        "하이닉스_2024.md", 2024, "하이닉스", "직원 1인당 당기순이익", "1인당 당기순이익")

    # body_text (fresh paragraphs in the same reports as discarded)
    replacements["case_082"] = _gen_body_text(client, "현대자동차_2023.md", "현대자동차", 2023, used["현대자동차_2023.md"])
    replacements["case_087"] = _gen_body_text(client, "삼성전자_2025.md", "삼성전자", 2025, used["삼성전자_2025.md"])

    # apply (keep id, validate)
    for cid, new in replacements.items():
        new["id"] = cid
        EvalCase.model_validate({**new, "gold_failure_type": "correct", **{k: v for k, v in new.items() if k != "needs_review"}})
        byid[cid] = new
        print(f"replaced {cid} [{new['slice']}]: {new['question']}")
        if new["slice"] == "body_text":
            print(f"    A: {json.dumps(new['expected_answer'], ensure_ascii=False)}")

    out = [byid[c["id"]] for c in rows]  # preserve original order
    DRAFT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out) + "\n", encoding="utf-8")
    print(f"\nwrote {len(out)} cases -> {DRAFT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
