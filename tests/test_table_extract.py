"""Unit tests for the pure table-normalization logic (no PDF / pdfplumber needed)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.table_extract import (
    collect_period_years,
    find_unit_captions,
    table_to_markdown,
)


def test_flattens_multi_level_header_and_fills_merged_header_cells():
    # Top header row: "구분" then a merged "연결" spanning two year columns (merge -> None).
    # Second header row: the year labels.
    rows = [
        ["구분", "연결", None],
        [None, "2024", "2023"],
        ["매출액", "300,870", "258,935"],
    ]
    md = table_to_markdown(rows)
    header_line = md.splitlines()[0]
    assert header_line == "| 구분 | 연결 / 2024 | 연결 / 2023 |"
    assert "| 매출액 | 300,870 | 258,935 |" in md


def test_fills_vertically_merged_row_labels():
    rows = [
        ["항목", "2024"],
        ["유동자산", "100"],
        [None, "200"],  # vertically merged label -> forward-filled
    ]
    md = table_to_markdown(rows)
    assert "| 유동자산 | 200 |" in md


def test_unit_note_is_attached_above_table():
    rows = [["구분", "2024"], ["매출", "100"]]
    md = table_to_markdown(rows, unit_note="(단위: 백만원)")
    assert md.splitlines()[0] == "_(단위: 백만원)_"


def test_escapes_pipe_in_cell():
    rows = [["a|b", "2024"], ["x", "1"]]
    md = table_to_markdown(rows)
    assert "a\\|b" in md


def test_empty_table_returns_empty_string():
    assert table_to_markdown([[None, None], ["", ""]]) == ""


def test_find_unit_captions_variants():
    text = "재무상태표 (단위 : 백만원)\n... (단위: 억원)"
    captions = find_unit_captions(text)
    assert "(단위: 백만원)" in captions
    assert "(단위: 억원)" in captions


def test_unit_caption_keeps_multiple_units_after_comma():
    # Regression: "(단위 : 억원, %)" was truncated to "(단위: 억원,)".
    assert find_unit_captions("(단위 : 억원, %)") == ["(단위: 억원, %)"]


def test_collect_period_years_maps_periods():
    text = (
        "제 55 기 2023.01.01 부터 2023.12.31 까지\n"
        "제 54 기 2022.01.01 부터 2022.12.31 까지\n"
        "제 55 기말 2023.12.31 현재"
    )
    mapping: dict[str, int] = {}
    collect_period_years(text, mapping)
    assert mapping == {"제55기": 2023, "제54기": 2022}


def test_header_period_labels_annotated_with_year():
    rows = [["구 분", "제 55 기", "제 54 기"], ["영업이익", "100", "200"]]
    md = table_to_markdown(rows, period_year={"제55기": 2023, "제54기": 2022})
    header_line = md.splitlines()[0]
    assert header_line == "| 구 분 | 제 55 기(2023) | 제 54 기(2022) |"


def test_period_label_unmapped_is_left_unchanged():
    rows = [["구 분", "제 99 기"], ["영업이익", "100"]]
    md = table_to_markdown(rows, period_year={"제55기": 2023})
    assert "제 99 기" in md and "제 99 기(" not in md


def test_section_label_only_row_not_merged_into_period_header():
    # Balance-sheet shape: year header row, then a '자산' section-label-only row.
    rows = [
        ["", "제 55 기", "제 54 기"],
        ["자산", "", ""],
        ["유동자산", "100", "200"],
    ]
    md = table_to_markdown(rows, period_year={"제55기": 2023, "제54기": 2022})
    assert md.splitlines()[0] == "|  | 제 55 기(2023) | 제 54 기(2022) |"
    assert "/ 자산" not in md  # section label not leaked into header
    assert "| 자산 |" in md  # kept as a separate body/divider row
    assert "| 유동자산 | 100 | 200 |" in md  # data row untouched
