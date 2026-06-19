"""Unit tests for the deterministic scorer (Phase 2.1).

Unit normalization is the highest-risk part — a conversion bug flips scores — so
the equivalence and negative cases are tested first and explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.evaluator.scorer import (
    _UNIT_WON,
    extract_amounts,
    parse_signed_number,
    score_case,
)


# --- unit normalization (★ highest priority) -------------------------------- #

def test_amount_equivalence_jo_eok_vs_baekman():
    won_a = extract_amounts("15조 4,871억원")
    won_b = extract_amounts("15,487,100백만원")
    gold_won = 15_487_100 * _UNIT_WON["백만원"]
    assert won_a == [pytest.approx(gold_won)]
    assert won_b == [pytest.approx(gold_won)]
    assert won_a[0] == pytest.approx(won_b[0]) == pytest.approx(gold_won)


def test_negative_paren_and_triangle_equal():
    assert parse_signed_number("(533,448)") == -533448
    assert parse_signed_number("△533,448") == -533448
    assert parse_signed_number("533,448") == 533448


def test_two_separate_amounts_not_merged():
    # non-descending scale separated -> two values, not summed
    assert extract_amounts("삼성전자 301조원, 하이닉스 66조원") == [pytest.approx(301e12), pytest.approx(66e12)]


# --- numeric scoring -------------------------------------------------------- #

def _numeric_case(value, unit="백만원"):
    return {"id": "c", "slice": "table_value", "answer_schema": "numeric",
            "expected_answer": {"value": value, "unit": unit, "basis": "연결", "period": "x"}}


def test_numeric_passes_comma_and_unit_notation():
    case = _numeric_case(15_487_100)
    assert score_case(case, "약 15조 4,871억원입니다.")["correct"] is True
    assert score_case(case, "15,487,100백만원")["correct"] is True
    assert score_case(case, "15487100")["correct"] is True  # bare -> gold unit


def test_numeric_value_mismatch_incorrect():
    # clearly >0.1% off -> incorrect (sub-0.1% rounding handled in tolerance test)
    assert score_case(_numeric_case(15_487_100), "14,000,000백만원")["correct"] is False


def test_table_value_relative_tolerance():
    # 억-unit rounding within ±0.1% -> correct (case_034 shape)
    assert score_case(_numeric_case(14_239_592), "14조 2,396억원")["correct"] is True
    # 0.139% off -> incorrect (case_031 shape)
    assert score_case(_numeric_case(23_467_319), "23.5조")["correct"] is False
    # exact -> correct
    assert score_case(_numeric_case(23_467_319), "23,467,319백만원")["correct"] is True


def test_numeric_refusal_on_answerable_incorrect():
    r = score_case(_numeric_case(15_487_100), "정보 없음")
    assert r["correct"] is False and r["score_detail"]["refused"] is True


def test_derived_percent_tolerance():
    case = {"id": "g", "slice": "numeric_reasoning", "answer_schema": "numeric",
            "expected_answer": {"value": -72.17, "unit": "%", "basis": "연결", "period": "x"}}
    assert score_case(case, "전년 대비 약 -72.2% 입니다")["correct"] is True   # within 0.1
    assert score_case(case, "약 -72.0%")["correct"] is False                   # 0.17 > 0.1


# --- comparison ------------------------------------------------------------- #

def _cmp_case():
    return {"id": "cmp", "slice": "numeric_reasoning", "answer_schema": "comparison",
            "expected_answer": {"companies": {
                "삼성전자": {"value": 300_870_903, "unit": "백만원"},
                "하이닉스": {"value": 66_192_960, "unit": "백만원"}},
                "comparison": "삼성전자 > 하이닉스"}}


def test_comparison_both_correct():
    a = "삼성전자 300,870,903백만원, 하이닉스 66,192,960백만원으로 삼성전자가 더 큽니다."
    assert score_case(_cmp_case(), a)["correct"] is True


def test_comparison_one_value_wrong_is_incorrect_not_ambiguous():
    a = "삼성전자 300,870,903백만원, 하이닉스 99,999,999백만원."
    r = score_case(_cmp_case(), a)
    assert r["correct"] is False
    assert r["score_detail"]["comparison_parse"] == "ok"  # parsed fine, just wrong


def test_comparison_summary_answer_is_ambiguous():
    a = "삼성전자가 하이닉스보다 매출이 더 많습니다."  # no per-company numbers
    r = score_case(_cmp_case(), a)
    assert r["correct"] is False
    assert r["score_detail"]["comparison_parse"] == "ambiguous"


def test_comparison_difference_answer_is_ambiguous():
    a = "삼성전자와 하이닉스의 매출 차이는 약 234,677,943백만원입니다."
    r = score_case(_cmp_case(), a)
    assert r["correct"] is False
    assert r["score_detail"]["comparison_parse"] == "ambiguous"


def test_comparison_company_declined_distinguished_from_ambiguous():
    a = "삼성전자 300,870,903백만원, 하이닉스 정보 없음."
    r = score_case(_cmp_case(), a)
    assert r["correct"] is False
    assert r["score_detail"]["comparison_parse"] == "company_declined"


# --- no_answer -------------------------------------------------------------- #

def _na_case():
    return {"id": "na", "slice": "no_answer", "answer_schema": "no_answer",
            "expected_answer": {"sentinel": "정보 없음"}}


def test_no_answer_refusal_correct():
    r = score_case(_na_case(), "정보 없음")
    assert r["correct"] is True and r["score_detail"]["over_answer"] is False


def test_no_answer_fabrication_is_over_answer():
    r = score_case(_na_case(), "2026년 매출은 약 350조원으로 전망됩니다.")
    assert r["correct"] is False and r["score_detail"]["over_answer"] is True


def test_body_text_deferred_to_judge():
    case = {"id": "b", "slice": "body_text", "answer_schema": "text",
            "expected_answer": {"key_points": ["x"]}}
    r = score_case(case, "어떤 답")
    assert r["correct"] is None and r["score_detail"]["deferred_to_judge"] is True
