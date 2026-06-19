"""Phase 1.2a — build a 100-case eval-set DRAFT (data/eval_draft.jsonl).

Generation (mixed, per ticket):
  * table_value (40), numeric_reasoning single growth (12), comparison (8),
    no_answer (15)  -> deterministic parsing of extracted Markdown (no LLM).
  * body_text (25)                                  -> LLM (OpenAI gpt-4o-mini).

Every case is a DRAFT: needs_review=True, gold_failure_type="correct".
Numeric answers are taken verbatim from parsed cells; source_ref pins the exact
file / table / row so a reviewer can verify each in one line.

Only 연결(consolidated) statements are used for numeric cases, so `basis` is never
guessed (the 별도 section headings do not contain '별도').
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.env import require_openai_key
from app.rag.corpus_md import (
    MdTable,
    clean_label,
    load_paragraphs,
    load_tables,
    parse_number,
    period_label,
    year_columns,
)

EXTRACTED_DIR = Path("data/corpus/extracted")
OUT_PATH = Path("data/eval_draft.jsonl")
SEED = 0
LLM_MODEL = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_eval_set")

# Canonical item -> set of accepted (space-removed) row labels, per statement type.
INCOME_ITEMS = {
    "매출액": {"매출액", "영업수익", "수익(매출액)", "매출"},
    "영업이익": {"영업이익"},
    "당기순이익": {"당기순이익", "당기순이익(손실)", "연결당기순이익"},
    "매출총이익": {"매출총이익"},
}
BALANCE_ITEMS = {
    "자산총계": {"자산총계"},
    "부채총계": {"부채총계"},
    "자본총계": {"자본총계"},
    "유동자산": {"유동자산"},
    "유동부채": {"유동부채"},
}
CASHFLOW_ITEMS = {
    "영업활동현금흐름": {"영업활동현금흐름", "영업활동으로인한현금흐름", "영업활동순현금흐름"},
    "투자활동현금흐름": {"투자활동현금흐름", "투자활동으로인한현금흐름", "투자활동순현금흐름"},
    "재무활동현금흐름": {"재무활동현금흐름", "재무활동으로인한현금흐름", "재무활동순현금흐름"},
}
BODY_SECTION_KEYWORDS = ("사업의 개요", "사업의 내용", "위험", "연구개발", "주요 계약",
                         "주요계약", "시장", "경쟁", "영업개황")


def statement_type(title: str) -> str | None:
    """Map a consolidated statement title to income/balance/cashflow, else None.

    '손익계산서' matches both '연결 손익계산서' and '연결 포괄손익계산서': some issuers
    (e.g. 하이닉스) use a single statement of comprehensive income that carries
    매출액/영업이익. For issuers with both, document order + per-item dedup keeps the
    plain income statement's value (it appears first).
    """
    if "연결" not in title:
        return None
    if "손익계산서" in title:
        return "income"
    if "재무상태표" in title:
        return "balance"
    if "현금흐름표" in title:
        return "cashflow"
    return None


def _items_for(stype: str) -> dict[str, set[str]]:
    return {"income": INCOME_ITEMS, "balance": BALANCE_ITEMS, "cashflow": CASHFLOW_ITEMS}[stype]


def _match_item(label: str, items: dict[str, set[str]]) -> str | None:
    base = clean_label(label)
    candidates = {
        base.replace(" ", ""),
        re.sub(r"\([^)]*\)", "", base).replace(" ", ""),  # also strip (손실)/(매출액) etc.
    }
    for canonical, synonyms in items.items():
        if candidates & synonyms:
            return canonical
    return None


def _gold_snippet(table: MdTable, row: list[str]) -> str:
    return ("| " + " | ".join(table.headers) + " |\n"
            + "| " + " | ".join(row) + " |")


# --------------------------------------------------------------------------- #
# Indexing: pull (company, year, item) -> record from 연결 statements
# --------------------------------------------------------------------------- #

class Record:
    def __init__(self, table: MdTable, stype: str, item: str, row: list[str],
                 col: int, year: int):
        self.table = table
        self.stype = stype
        self.item = item
        self.row = row
        self.col = col
        self.value = parse_number(row[col])
        self.year = year
        self.unit = table.unit or ""
        self.period = period_label(table.headers[col], year)
        self.label = clean_label(row[0])

    def source_ref_str(self) -> str:
        return (f"{self.table.file}#Table {self.table.table_id}"
                f"#{self.label} [{self.table.headers[self.col]}]")


def index_consolidated(tables: list[MdTable], report_year: int) -> list[Record]:
    """Records for the report-year column of every consolidated statement."""
    out: list[Record] = []
    seen: set[tuple[str, str]] = set()  # (stype, item), document order keeps first
    for tb in tables:
        stype = statement_type(tb.title)
        if not stype:
            continue
        yc = year_columns(tb.headers)
        if report_year not in yc:
            continue
        col = yc[report_year]
        items = _items_for(stype)
        for row in tb.rows:
            if col >= len(row):
                continue
            canonical = _match_item(row[0], items)
            if not canonical or (stype, canonical) in seen:
                continue
            rec = Record(tb, stype, canonical, row, col, report_year)
            if rec.value is None:
                continue
            seen.add((stype, canonical))
            out.append(rec)
    return out


def main() -> int:
    rng = random.Random(SEED)
    logger.info("seed=%d  input=%s  output=%s  llm=%s",
                SEED, EXTRACTED_DIR.resolve(), OUT_PATH, LLM_MODEL)

    files = sorted(EXTRACTED_DIR.glob("*.md"))
    if not files:
        logger.error("no extracted .md files in %s", EXTRACTED_DIR)
        return 1

    # Per-file consolidated index (keyed by report year = file year).
    per_file: dict[str, list[Record]] = {}
    company_of: dict[str, str] = {}
    year_of: dict[str, int] = {}
    for f in files:
        tables = load_tables(f)
        company, year = tables[0].company, tables[0].report_year
        fn = nfc(f.name)
        company_of[fn] = company
        year_of[fn] = year
        per_file[fn] = index_consolidated(tables, year)
        logger.info("%s: %d consolidated records", fn, len(per_file[fn]))

    cases: list[dict] = []

    # ---- table_value (40): round-robin across files for balance ----
    pools = {fn: list(recs) for fn, recs in per_file.items()}
    for recs in pools.values():
        rng.shuffle(recs)
    tv: list[Record] = []
    while len(tv) < 40 and any(pools.values()):
        for fn in sorted(pools):
            if pools[fn]:
                tv.append(pools[fn].pop())
                if len(tv) == 40:
                    break
    for r in tv:
        cases.append(_numeric_case(r, slice_="table_value"))

    # ---- numeric_reasoning single (12): YoY growth on income items ----
    growth_pools: dict[str, list[dict]] = {fn: [] for fn in per_file}
    for fn in per_file:
        tables = load_tables(EXTRACTED_DIR / fn)
        yr = year_of[fn]
        for tb in tables:
            if statement_type(tb.title) != "income":
                continue
            yc = year_columns(tb.headers)
            if yr not in yc or (yr - 1) not in yc:
                continue
            for row in tb.rows:
                canonical = _match_item(row[0], INCOME_ITEMS)
                if not canonical:
                    continue
                cur, prev = parse_number(row[yc[yr]]), parse_number(row[yc[yr - 1]])
                if cur is None or prev is None or prev <= 0:
                    continue  # base year must be positive, else growth % is meaningless
                growth = round((cur - prev) / prev * 100, 2)
                growth_pools[fn].append(_growth_case(tb, row, canonical, yr, growth))
            break  # one income table per file is enough
    growth: list[dict] = []
    while len(growth) < 12 and any(growth_pools.values()):
        for fn in sorted(growth_pools):
            if growth_pools[fn]:
                growth.append(growth_pools[fn].pop())
                if len(growth) == 12:
                    break
    cases.extend(growth)

    # ---- comparison (8): 6 same-industry (삼성↔하이닉스) + 2 cross (현대차) ----
    cases.extend(_comparison_cases(per_file, company_of, year_of))

    # ---- no_answer (15) ----
    cases.extend(_no_answer_cases(files, company_of, year_of, rng))

    # ---- body_text (25) via LLM ----
    cases.extend(_body_text_cases(files, rng))

    # ---- assign ids, write ----
    for i, c in enumerate(cases, start=1):
        c["id"] = f"case_{i:03d}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("wrote %d cases -> %s", len(cases), OUT_PATH)

    _print_summary(cases)
    return 0


# --------------------------------------------------------------------------- #
# Case builders
# --------------------------------------------------------------------------- #

def _numeric_case(r: Record, slice_: str) -> dict:
    return {
        "company": r.table.company,
        "source_doc": r.table.file,
        "fiscal_year": r.year,
        "question": f"{r.year}년 연결 {r.item}은?",
        "contexts": [_gold_snippet(r.table, r.row)],
        "answer_schema": "numeric",
        "expected_answer": {"value": r.value, "unit": r.unit, "basis": "연결",
                            "period": r.period},
        "answer_type": "answerable",
        "slice": slice_,
        "gold_failure_type": "correct",
        "source_ref": r.source_ref_str(),
        "needs_review": True,
    }


def _growth_case(tb: MdTable, row: list[str], item: str, year: int, growth: float) -> dict:
    return {
        "company": tb.company,
        "source_doc": tb.file,
        "fiscal_year": year,
        "question": f"{year - 1}년 대비 {year}년 연결 {item} 증가율은? (%)",
        "contexts": [_gold_snippet(tb, row)],
        "answer_schema": "numeric",
        "expected_answer": {"value": growth, "unit": "%", "basis": "연결",
                            "period": f"{year - 1}→{year}"},
        "answer_type": "answerable",
        "slice": "numeric_reasoning",
        "gold_failure_type": "correct",
        "source_ref": f"{tb.file}#Table {tb.table_id}#{clean_label(row[0])} [{year - 1}↔{year}]",
        "needs_review": True,
    }


def _find(records: list[Record], item: str) -> Record | None:
    return next((r for r in records if r.item == item), None)


def _comparison_case(a: Record, b: Record, item: str, year: int) -> dict:
    comp = (f"{a.table.company} > {b.table.company}" if a.value > b.value
            else f"{b.table.company} > {a.table.company}" if b.value > a.value
            else f"{a.table.company} = {b.table.company}")
    return {
        "company": f"{a.table.company}|{b.table.company}",
        "source_doc": f"{a.table.file}|{b.table.file}",
        "fiscal_year": year,
        "question": f"{year}년 연결 {item} 기준, {a.table.company}와 {b.table.company} 중 큰 쪽과 두 값은?",
        "contexts": [_gold_snippet(a.table, a.row), _gold_snippet(b.table, b.row)],
        "answer_schema": "comparison",
        "expected_answer": {
            "companies": {
                a.table.company: {"value": a.value, "unit": a.unit},
                b.table.company: {"value": b.value, "unit": b.unit},
            },
            "comparison": comp,
        },
        "answer_type": "answerable",
        "slice": "numeric_reasoning",
        "gold_failure_type": "correct",
        "source_ref": f"{a.source_ref_str()} ; {b.source_ref_str()}",
        "needs_review": True,
    }


_CMP_ITEMS = ["매출액", "영업이익", "당기순이익", "매출총이익", "자산총계",
              "유동자산", "영업활동현금흐름"]
_CMP_YEARS = [2024, 2025, 2023]


def _comparison_cases(per_file, company_of, year_of) -> list[dict]:
    by_cy: dict[tuple[str, int], list[Record]] = {}
    for fn, recs in per_file.items():
        by_cy[(company_of[fn], year_of[fn])] = recs

    def rec(company, year, item):
        return _find(by_cy.get((company, year), []), item)

    def collect(c1: str, c2: str, want: int) -> list[dict]:
        """Pick `want` comparisons, round-robin over years so they spread across
        2023/2024/2025 instead of clustering on the year with the most items."""
        picked: list[dict] = []
        used: set[tuple[int, str]] = set()
        while len(picked) < want:
            progressed = False
            for yr in _CMP_YEARS:
                if len(picked) == want:
                    break
                for item in _CMP_ITEMS:
                    if (yr, item) in used:
                        continue
                    a, b = rec(c1, yr, item), rec(c2, yr, item)
                    if a and b:
                        picked.append(_comparison_case(a, b, item, yr))
                        used.add((yr, item))
                        progressed = True
                        break
            if not progressed:
                break
        return picked

    out = collect("삼성전자", "하이닉스", 6)  # same-industry
    out += collect("삼성전자", "현대자동차", 1)  # cross-industry
    out += collect("하이닉스", "현대자동차", 1)  # cross-industry
    if len(out) != 8:
        logger.warning("comparison cases generated: %d (expected 8)", len(out))
    return out


def _no_answer_cases(files, company_of, year_of, rng) -> list[dict]:
    names = [nfc(f.name) for f in files]
    out: list[dict] = []

    def case(file, core):
        # Prefix company + report year (same format as other slices), so the
        # question is self-contained. For future-year traps the prefix shows the
        # report year while the body asks about a later year.
        return {
            "company": company_of[file],
            "source_doc": file,
            "fiscal_year": year_of[file],
            "question": f"{company_of[file]} {year_of[file]}년 보고서 기준, {core}",
            "contexts": [],
            "answer_schema": "no_answer",
            "expected_answer": {"sentinel": "정보 없음"},
            "answer_type": "unanswerable",
            "slice": "no_answer",
            "gold_failure_type": "correct",
            "source_ref": file,
            "needs_review": True,
        }

    # 다른 연도 (5): years/items/companies varied; report year < question year
    year_traps = [
        ("삼성전자_2023.md", 2026, "매출액"),
        ("하이닉스_2024.md", 2027, "영업이익"),
        ("현대자동차_2025.md", 2026, "당기순이익"),
        ("삼성전자_2025.md", 2027, "매출액"),
        ("하이닉스_2023.md", 2026, "영업이익"),
    ]
    for fn, yr, item in year_traps:
        assert year_of[fn] < yr, f"future-year trap broken: {fn} ({year_of[fn]}) !< {yr}"
        out.append(case(fn, f"{yr}년 연결 {item}은 얼마인가?"))
    # 다른 주체 (5): entity absent from corpus; 1 comparison trap
    entities = ["TSMC", "Apple", "Intel", "도요타"]
    for fn, ent in zip(names[5:9], entities):
        out.append(case(fn, f"{ent}의 2024년 매출은 얼마인가?"))
    out.append(case("삼성전자_2024.md",
                    "TSMC의 2024년 연결 매출과 비교하면 어느 쪽이 더 큰가?"))  # comparison trap
    # 없는 항목 (5): per-capita metrics that reports don't disclose (grep-verified
    # absent; reports carry per-capita 급여 only, not per-capita profit/revenue).
    missing = ["직원 1인당 매출액", "직원 1인당 영업이익", "직원 1인당 당기순이익",
               "직원 1인당 영업이익", "직원 1인당 당기순이익"]
    for fn, item in zip(names[:5], missing):
        out.append(case(fn, f"보고된 {item}은 얼마인가?"))
    return out


def _body_text_cases(files, rng) -> list[dict]:
    from openai import OpenAI

    require_openai_key()
    client = OpenAI()

    # gather candidate paragraphs (section keywords, distribute ~3/file)
    candidates: list = []
    per_file_paras: dict[str, list] = {}
    for f in files:
        paras = [p for p in load_paragraphs(f)
                 if any(k in p.text[:120] for k in BODY_SECTION_KEYWORDS)]
        if not paras:
            paras = load_paragraphs(f)
        rng.shuffle(paras)
        per_file_paras[nfc(f.name)] = paras

    # round-robin to 25
    while len(candidates) < 25 and any(per_file_paras.values()):
        for fn in sorted(per_file_paras):
            if per_file_paras[fn]:
                candidates.append(per_file_paras[fn].pop())
                if len(candidates) == 25:
                    break

    out: list[dict] = []
    fails = 0
    for p in candidates:
        core, key_points = _llm_question(client, p.company, p.report_year, p.text[:1800])
        if core is None:
            fails += 1
            core = f"이 문단(p.{p.page})이 설명하는 핵심 사실은 무엇인가?"
            key_points = ["[LLM 생성 실패 — 검수 시 작성 필요]"]
        # Force company + report year into the question (deterministic, like the
        # numeric slices), so the question is uniquely answerable on its own.
        question = f"{p.company} {p.report_year}년 보고서 기준, {core}"
        out.append({
            "company": p.company,
            "source_doc": p.file,
            "fiscal_year": p.report_year,
            "question": question,
            "contexts": [p.text],
            "answer_schema": "text",
            "expected_answer": {"key_points": key_points},
            "answer_type": "answerable",
            "slice": "body_text",
            "gold_failure_type": "correct",
            "source_ref": f"{p.file}#p.{p.page}",
            "needs_review": True,
        })
    if fails:
        logger.warning("body_text: %d LLM generations fell back to placeholder", fails)
    return out


def _llm_question(client, company: str, year: int, paragraph: str) -> tuple[str | None, list[str]]:
    prompt = (
        f"다음은 {company}의 {year}년 사업보고서 본문 한 문단이다. "
        "이 문단만 근거로 답이 하나로 정해지는 사실 질문 1개와, 그 질문이 직접 "
        "묻는 것에만 답하는 간결한 정답 핵심 포인트를 만들어라.\n"
        "규칙:\n"
        "- 질문에는 회사명·연도를 넣지 마라(시스템이 접두사로 붙인다). '당사' 같은 표현도 쓰지 마라.\n"
        "- 단일 사실을 묻는 질문이면 key_points는 1개로. 최대 3개.\n"
        "- 질문과 무관한 배경·옆 문단 내용·부연 설명을 넣지 마라.\n"
        "- 반드시 이 JSON 형식으로만 답하라: "
        '{"question": "...", "key_points": ["..."]}\n\n문단:\n' + paragraph
    )
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content)
            q = str(data["question"]).strip()
            kp = [str(x).strip() for x in data["key_points"] if str(x).strip()]
            if q and kp:
                return q, kp[:3]
        except Exception as exc:  # noqa: BLE001 - graceful fallback per CLAUDE.md
            logger.debug("LLM gen failed: %s", exc)
    return None, []


# --------------------------------------------------------------------------- #
# Self-validation summary
# --------------------------------------------------------------------------- #

FORBIDDEN_REF = ("명세", "종속기업", "관계기업", "출자", "타법인")


def _print_summary(cases: list[dict]) -> None:
    from app.schemas import EvalCase

    logger.info("===== DRAFT SUMMARY =====")
    logger.info("total cases: %d", len(cases))

    slices = Counter(c["slice"] for c in cases)
    expected = {"table_value": 40, "numeric_reasoning": 20, "body_text": 25, "no_answer": 15}
    for s, n in expected.items():
        mark = "OK" if slices.get(s) == n else "MISMATCH"
        logger.info("  slice %-18s %d / %d  [%s]", s, slices.get(s, 0), n, mark)
    comparison = sum(1 for c in cases if c["answer_schema"] == "comparison")
    logger.info("  numeric_reasoning comparison: %d / 8 [%s]",
                comparison, "OK" if comparison == 8 else "MISMATCH")

    # company / year distribution (single-company cases only)
    comp = Counter(c["company"] for c in cases if "|" not in c["company"])
    logger.info("company distribution (excl. comparison): %s", dict(comp))
    years = Counter(c["fiscal_year"] for c in cases)
    logger.info("fiscal_year distribution: %s", dict(sorted(years.items())))

    # forbidden source_ref check
    bad = [c["id"] for c in cases if any(k in c["source_ref"] for k in FORBIDDEN_REF)]
    logger.info("forbidden-table refs: %d  [%s]", len(bad), "OK" if not bad else bad)

    # body_text: company+year stated in question AND matching the source report
    bt = [c for c in cases if c["slice"] == "body_text"]
    bad_bt = [c["id"] for c in bt
              if not (c["company"] in c["question"]
                      and f"{c['fiscal_year']}년" in c["question"]
                      and c["company"] in c["source_doc"]
                      and str(c["fiscal_year"]) in c["source_doc"])]
    logger.info("body_text 회사+연도 명시 & source 일치: %d/%d  [%s]",
                len(bt) - len(bad_bt), len(bt), "OK" if not bad_bt else bad_bt)

    na = [c for c in cases if c["slice"] == "no_answer"]
    bad_na = [c["id"] for c in na
              if not (c["company"] in c["question"]
                      and f"{c['fiscal_year']}년" in c["question"]
                      and c["company"] in c["source_doc"]
                      and str(c["fiscal_year"]) in c["source_doc"])]
    logger.info("no_answer 회사+연도(보고서) 명시 & source 일치: %d/%d  [%s]",
                len(na) - len(bad_na), len(na), "OK" if not bad_na else bad_na)

    # schema validation
    errors = 0
    for c in cases:
        try:
            EvalCase.model_validate(c)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.warning("schema FAIL %s: %s", c.get("id"), exc)
    logger.info("schema validation: %d/%d valid", len(cases) - errors, len(cases))
    logger.info("needs_review True: %d/%d", sum(c["needs_review"] for c in cases), len(cases))


if __name__ == "__main__":
    raise SystemExit(main())
