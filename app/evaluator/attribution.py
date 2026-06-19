"""Failure attribution (Phase 2.3) — assign a single primary_failure per case.

MVP modes: correct / retrieval_miss / table_value_error / hallucination.

retrieval_miss is decided deterministically from logs (no judge): does the gold
evidence (source_ref's table_id / page) appear in retrieved_chunks? Matching is
table-level (any fragment of the gold table counts). Comparison needs BOTH gold
tables; either missing -> retrieval_miss.

When the gold evidence WAS retrieved but the answer is wrong:
  * model value matches some other number in the retrieved gold table -> table_value_error
  * otherwise -> hallucination
A borderline numeric case (model value near, but not within, a gold cell) gets one
optional judge assist.
"""

from __future__ import annotations

import re
import unicodedata

from app.config import RagConfig
from app.evaluator.scorer import _candidates_in_gold_unit, parse_signed_number

_TABLE_REF = re.compile(r"^(?P<file>.*?)#Table\s+(?P<tid>[^#]+?)#")
_PAGE_REF = re.compile(r"^(?P<file>.*?)#p\.(?P<page>\d+)")
_NUMTOK = re.compile(r"(?:△|▲|[-−])?\(?\d[\d,]{2,}(?:\.\d+)?\)?")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def gold_keys(case: dict) -> list[tuple]:
    """Gold evidence keys from source_ref. ('table', file, tid) or ('page', file, page)."""
    keys: list[tuple] = []
    for ref in _nfc(case.get("source_ref", "")).split(" ; "):
        ref = ref.strip()
        m = _TABLE_REF.match(ref)
        if m:
            keys.append(("table", _nfc(m.group("file")), _nfc(m.group("tid").strip())))
            continue
        m = _PAGE_REF.match(ref)
        if m:
            keys.append(("page", _nfc(m.group("file")), int(m.group("page"))))
    return keys


def retrieved_keys(run_record: dict) -> set[tuple]:
    keys: set[tuple] = set()
    for c in run_record.get("retrieved_chunks", []):
        meta = c.get("metadata", {})
        f = _nfc(meta.get("source_file", ""))
        if meta.get("is_table") and meta.get("table_id"):
            keys.add(("table", f, _nfc(meta["table_id"])))
        if meta.get("page") is not None:
            keys.add(("page", f, meta["page"]))
    return keys


def is_retrieval_miss(case: dict, run_record: dict) -> bool:
    golds = gold_keys(case)
    if not golds:  # no_answer has no gold table -> retrieval_miss N/A
        return False
    have = retrieved_keys(run_record)
    return not all(g in have for g in golds)  # any gold table missing -> miss


def _numbers_in(text: str) -> set[float]:
    out: set[float] = set()
    for tok in _NUMTOK.findall(text or ""):
        v = parse_signed_number(tok)
        if v is not None:
            out.add(v)
    return out


def _gold_evidence_numbers(case: dict, run_record: dict) -> set[float]:
    """Numbers in the retrieved chunks that belong to the gold table(s)."""
    golds = set(gold_keys(case))
    nums: set[float] = set()
    for c in run_record.get("retrieved_chunks", []):
        meta = c.get("metadata", {})
        f = _nfc(meta.get("source_file", ""))
        key_t = ("table", f, _nfc(meta.get("table_id", "")))
        key_p = ("page", f, meta.get("page"))
        if key_t in golds or key_p in golds:
            nums |= _numbers_in(c.get("text", ""))
    return nums


def _model_values(case: dict, scored: dict, run_record: dict) -> list[float]:
    """Model's numeric values in the gold unit (reusing scorer parse)."""
    schema = case["answer_schema"]
    detail = scored.get("score_detail", {})
    if schema == "numeric":
        return detail.get("parsed_in_gold_unit") or _candidates_in_gold_unit(
            run_record.get("answer") or "", case["expected_answer"].get("unit", "원"))
    if schema == "comparison":
        vals: list[float] = []
        for v in (detail.get("per_company") or {}).values():
            if isinstance(v, list):
                vals.extend(v)
        return vals
    return []


def _value_error_or_hallucination(case, scored, run_record, config, client) -> tuple[str, dict]:
    gold_nums = _gold_evidence_numbers(case, run_record)
    model_vals = _model_values(case, scored, run_record)
    gold_answer = case["expected_answer"].get("value")

    def near(a, b):
        return abs(a - b) <= 0.001 * abs(b) if b else a == b

    misread = any(near(mv, gn) for mv in model_vals for gn in gold_nums
                  if not (gold_answer is not None and near(mv, gold_answer)))
    if misread:
        return "table_value_error", {"reason": "model value matches a non-answer cell in gold table"}

    # borderline (model value close-ish to a gold cell but not within 0.1%): judge assist
    borderline = any(0.001 * abs(gn) < abs(mv - gn) <= 0.02 * abs(gn)
                     for mv in model_vals for gn in gold_nums if gn)
    if borderline and client is not None:
        verdict = _judge_assist(case, run_record, config, client)
        return verdict, {"reason": "judge_assist", "judge": verdict}

    return "hallucination", {"reason": "model value not found in retrieved gold table"}


def _judge_assist(case, run_record, config, client) -> str:
    """One judge call to disambiguate table_value_error vs hallucination."""
    import json

    prompt = (
        "표 기반 QA의 오답을 분류하라. 모델이 표의 다른 셀(연도/항목)을 잘못 읽었으면 "
        "'table_value_error', 표에 없는 값을 지어냈으면 'hallucination'.\n"
        f"질문: {case['question']}\n정답: {json.dumps(case['expected_answer'], ensure_ascii=False)}\n"
        f"모델 답: {run_record.get('answer')}\n"
        'JSON: {"label": "table_value_error" 또는 "hallucination"}'
    )
    try:
        resp = client.chat.completions.create(
            model=config.judge_model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        label = json.loads(resp.choices[0].message.content).get("label")
        return label if label in ("table_value_error", "hallucination") else "hallucination"
    except Exception:  # noqa: BLE001
        return "hallucination"


def attribute(case: dict, correct: bool | None, scored: dict, run_record: dict,
              config: RagConfig, client=None) -> dict:
    """Assign primary_failure. `correct` is the final verdict (judge for body_text)."""
    base = {"id": case["id"], "slice": case["slice"]}
    detail: dict = {}

    if correct:
        return {**base, "primary_failure": "correct", "attribution_detail": detail}

    # no_answer over_answer -> hallucination (keep the over_answer flag)
    if case["answer_schema"] == "no_answer":
        return {**base, "primary_failure": "hallucination",
                "attribution_detail": {"over_answer": scored.get("score_detail", {}).get("over_answer", True)}}

    if is_retrieval_miss(case, run_record):
        return {**base, "primary_failure": "retrieval_miss",
                "attribution_detail": {"gold_keys": gold_keys(case),
                                       "retrieved_tables": sorted(
                                           f"{f}:{t}" for kind, f, t in retrieved_keys(run_record) if kind == "table")[:10]}}

    if case["answer_schema"] == "text":  # body_text retrieved but wrong -> no table to misread
        return {**base, "primary_failure": "hallucination",
                "attribution_detail": {"reason": "gold paragraph retrieved but answer judged incorrect"}}

    label, d = _value_error_or_hallucination(case, scored, run_record, config, client)
    return {**base, "primary_failure": label, "attribution_detail": d}
