"""Deterministic slice-aware scoring (Phase 2.1). No LLM here.

numeric / comparison / no_answer are scored by parsing numbers from the model's
free-text Korean answer and comparing to the gold value. body_text is deferred to
the LLM judge (§2.2) — only the interface stub lives here.

Unit normalization is the highest-risk part (a conversion bug flips scores), so
parse_signed_number / extract_amounts are pure and unit-tested.
"""

from __future__ import annotations

import re

# Korean monetary/scale units -> multiplier in WON. Longest matched first.
_UNIT_WON: dict[str, float] = {
    "조원": 1e12, "조": 1e12,
    "십억원": 1e9,
    "억원": 1e8, "억": 1e8,
    "천만원": 1e7,
    "백만원": 1e6,
    "만원": 1e4, "만": 1e4,
    "천원": 1e3,
    "원": 1.0,
}
_UNIT_ALT = "|".join(sorted(_UNIT_WON, key=len, reverse=True))
_AMOUNT_RE = re.compile(rf"(△|▲|[-−])?\s*(\(?)\s*(\d[\d,]*(?:\.\d+)?)\s*(\)?)\s*({_UNIT_ALT})")
_PCT_RE = re.compile(r"(△|▲|[-−])?\s*(\(?)\s*(\d[\d,]*(?:\.\d+)?)\s*(\)?)\s*%")
_BARE_RE = re.compile(r"(△|▲|[-−])?\s*(\(?)\s*(\d[\d,]{3,}(?:\.\d+)?)\s*(\)?)")

_REFUSAL = ("정보 없음", "정보없음", "알 수 없", "확인할 수 없", "제공되지 않",
            "나와 있지 않", "찾을 수 없", "해당 정보가 없", "기재되어 있지 않",
            "포함되어 있지 않", "언급되어 있지 않", "보고서에 없")
_SUMMARY_KW = ("차이", "합계", "평균", "배 ", "총합", "합산", "더 많", "더 큼", "더 적", "더 작")


def parse_signed_number(token: str) -> float | None:
    """Parse a single signed number: '(533,448)' / '△533,448' -> -533448."""
    s = token.strip()
    neg = s[:1] in "△▲-−" or (s.startswith("(") and s.rstrip().endswith(")"))
    digits = re.sub(r"[△▲\-−()%,\s]", "", s)
    if not re.fullmatch(r"\d+(\.\d+)?", digits):
        return None
    val = float(digits)
    return -val if neg else val


def _signed(sign: str, lparen: str, num: str, rparen: str) -> float:
    val = float(num.replace(",", ""))
    if sign in ("△", "▲", "-", "−") or (lparen and rparen):
        val = -val
    return val


def extract_amounts(text: str) -> list[float]:
    """Won-valued amounts in text. Consecutive descending-scale tokens
    ('15조 4,871억') combine into one value; non-descending tokens are separate."""
    toks = [(_signed(m.group(1) or "", m.group(2), m.group(3), m.group(4)),
             _UNIT_WON[m.group(5)]) for m in _AMOUNT_RE.finditer(text)]
    groups: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for val, mult in toks:
        if cur and mult < cur[-1][1]:
            cur.append((val, mult))
        else:
            if cur:
                groups.append(cur)
            cur = [(val, mult)]
    if cur:
        groups.append(cur)
    return [sum(v * m for v, m in g) for g in groups]


def extract_percents(text: str) -> list[float]:
    return [_signed(m.group(1) or "", m.group(2), m.group(3), m.group(4))
            for m in _PCT_RE.finditer(text)]


def is_refusal(text: str) -> bool:
    return any(p in (text or "") for p in _REFUSAL)


def _candidates_in_gold_unit(text: str, unit: str) -> list[float]:
    """Model amounts expressed in gold's unit (amounts with units + bare numbers)."""
    mult = _UNIT_WON.get(unit, 1.0)
    cands = [w / mult for w in extract_amounts(text)]
    consumed = _AMOUNT_RE.sub(" ", text)  # bare numbers (no unit) -> assume gold unit
    for m in _BARE_RE.finditer(consumed):
        b = _signed(m.group(1) or "", m.group(2), m.group(3), m.group(4))
        cands.append(b)
    return cands


# --------------------------------------------------------------------------- #
# Slice scorers
# --------------------------------------------------------------------------- #

def score_numeric(case: dict, answer: str) -> dict:
    ea = case["expected_answer"]
    gold, unit = ea["value"], ea["unit"]
    detail: dict = {"gold": gold, "unit": unit}

    if is_refusal(answer):  # answerable but refused -> incorrect
        return {"correct": False, "score_detail": {**detail, "refused": True}}

    if unit == "%":
        cands = extract_percents(answer)
        detail["parsed_percents"] = cands
        correct = any(abs(c - gold) <= 0.1 for c in cands)  # ±0.1%p for derived values
    else:
        cands = _candidates_in_gold_unit(answer, unit)
        tol = 0.001 * abs(gold)  # ±0.1% relative: absorbs 억/조-unit rounding, not real errors
        detail["parsed_in_gold_unit"] = cands
        detail["rel_tol_pct"] = 0.1
        correct = any(abs(c - gold) <= tol for c in cands)
    return {"correct": correct, "score_detail": detail}


def _segments_by_company(answer: str, companies: list[str]) -> dict[str, str | None]:
    """Text segment attributed to each company (between its name and the next)."""
    pos = sorted((answer.find(c), c) for c in companies if answer.find(c) >= 0)
    segs: dict[str, str | None] = {c: None for c in companies}
    for i, (p, name) in enumerate(pos):
        end = pos[i + 1][0] if i + 1 < len(pos) else len(answer)
        segs[name] = answer[p + len(name):end]
    return segs


def score_comparison(case: dict, answer: str) -> dict:
    gold = case["expected_answer"]["companies"]  # {name: {value, unit}}
    segs = _segments_by_company(answer, list(gold))
    detail: dict = {"gold": {c: gold[c]["value"] for c in gold}, "per_company": {}}

    declined, missing = [], []
    per_vals: dict[str, list[float]] = {}
    for name, gv in gold.items():
        seg = segs[name]
        if seg is None:
            missing.append(name)
            detail["per_company"][name] = "not_mentioned"
            continue
        if is_refusal(seg):
            declined.append(name)
            detail["per_company"][name] = "declined"
            continue
        cands = _candidates_in_gold_unit(seg, gv["unit"])
        per_vals[name] = cands
        detail["per_company"][name] = cands or "no_number"
        if not cands:
            missing.append(name)

    # distinguish parse-failure (ambiguous) from a genuine wrong answer
    distinct = {tuple(round(v) for v in vs) for vs in per_vals.values()}
    summary_like = any(k in answer for k in _SUMMARY_KW)
    if declined:
        return {"correct": False, "score_detail": {**detail, "comparison_parse": "company_declined"}}
    if missing or (summary_like and len(distinct) < len(gold)) or (len(per_vals) == len(gold) and len(distinct) == 1):
        return {"correct": False, "score_detail": {**detail, "comparison_parse": "ambiguous"}}

    # both companies have distinct numbers -> both must match gold (direction follows)
    correct = all(any(round(c) == gold[name]["value"] for c in per_vals[name]) for name in gold)
    return {"correct": correct, "score_detail": {**detail, "comparison_parse": "ok"}}


def score_no_answer(case: dict, answer: str) -> dict:
    refused = is_refusal(answer) or not answer.strip()
    # over_answer: declined to refuse and asserted a concrete value
    over = not refused
    return {"correct": refused, "score_detail": {"over_answer": over}}


def score_case(case: dict, answer: str | None) -> dict:
    """Route a case + model answer to its scorer. body_text -> judge (deferred)."""
    base = {"id": case["id"], "slice": case["slice"]}
    if answer is None:
        return {**base, "correct": False, "score_detail": {"error": "no answer (pipeline failure)"}}

    schema = case["answer_schema"]
    if schema == "comparison":
        return {**base, **score_comparison(case, answer)}
    if schema == "numeric":
        return {**base, **score_numeric(case, answer)}
    if schema == "no_answer":
        return {**base, **score_no_answer(case, answer)}
    if schema == "text":  # body_text -> deferred to LLM judge (§2.2)
        return {**base, "correct": None, "score_detail": {"deferred_to_judge": True}}
    return {**base, "correct": False, "score_detail": {"error": f"unknown schema {schema}"}}
