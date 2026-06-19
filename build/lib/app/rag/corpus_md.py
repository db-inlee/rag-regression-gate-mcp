"""Parse extracted corpus Markdown (`data/corpus/extracted/*.md`) back into
structured tables and body paragraphs, for deterministic eval-set construction.

This reads the output of `table_extract.py`. Numbers are parsed from cells with
Korean accounting conventions (△ / parentheses = negative, thousands commas).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def _nfc(text: str) -> str:
    """Normalize to NFC. macOS filenames/PDF text can be NFD-decomposed, which
    breaks equality with NFC string literals in the codebase."""
    return unicodedata.normalize("NFC", text)

_YEAR_IN_HEADER = re.compile(r"\((\d{4})\)")
_PERIOD_NO = re.compile(r"제\s*(\d+)\s*기")
_FOOTNOTE = re.compile(r"\s*\(주[^)]*\)")
_UNIT_INLINE = re.compile(r"\(단위[^)]*\)")


@dataclass
class MdTable:
    company: str
    report_year: int  # from filename (the report's own year)
    page: int
    table_id: str  # e.g. "p82-1"
    title: str  # nearest preceding statement heading (for basis / type)
    unit: str | None  # e.g. "백만원"
    headers: list[str]
    rows: list[list[str]]
    file: str


@dataclass
class MdParagraph:
    company: str
    report_year: int
    page: int
    text: str
    file: str


# --------------------------------------------------------------------------- #
# Number / label helpers
# --------------------------------------------------------------------------- #

def parse_number(cell: str) -> int | float | None:
    """Parse a financial cell to a number, or None if it is not numeric.

    Handles △/▲ and (parentheses) as negative, thousands commas, trailing %.
    """
    s = (cell or "").strip()
    if s in {"", "-", "–", "—"}:
        return None
    neg = False
    if s[:1] in {"△", "▲", "−"}:
        neg, s = True, s[1:]
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = s.replace(",", "").replace(" ", "").rstrip("%")
    try:
        value: int | float = float(s) if "." in s else int(s)
    except ValueError:
        return None
    return -value if neg else value


def clean_label(label: str) -> str:
    """Strip footnote markers like '(주29)' and surrounding whitespace."""
    return _FOOTNOTE.sub("", label or "").strip()


def year_columns(headers: list[str]) -> dict[int, int]:
    """Map fiscal year -> column index, using the '(YYYY)' annotation in headers."""
    out: dict[int, int] = {}
    for idx, head in enumerate(headers):
        m = _YEAR_IN_HEADER.search(head)
        if m:
            out.setdefault(int(m.group(1)), idx)
    return out


def period_label(header_cell: str, year: int) -> str:
    """Build a period string like '2023(제55기)' from a header cell."""
    m = _PERIOD_NO.search(header_cell or "")
    return f"{year}(제{m.group(1)}기)" if m else str(year)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return set(line.strip()) <= {"|", "-", " "} and "-" in line


def _company_year(stem: str) -> tuple[str, int]:
    company, _, year = _nfc(stem).rpartition("_")
    return company, int(year)


def _recent_title(body: list[str]) -> str:
    keywords = ("포괄손익계산서", "손익계산서", "재무상태표", "현금흐름표",
                "요약재무", "요약 재무", "부문", "주요 제품", "주요제품")
    for line in reversed(body[-15:]):
        if any(k in line for k in keywords):
            return line
    return body[-1] if body else ""


def load_tables(path: Path) -> list[MdTable]:
    company, year = _company_year(path.stem)
    file = _nfc(path.name)
    lines = _nfc(path.read_text(encoding="utf-8")).splitlines()
    tables: list[MdTable] = []
    page = 0
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## p."):
            page = int(line[5:].strip().rstrip("."))
            body = []
            i += 1
            continue
        if line.startswith("### Table "):
            table_id = line[len("### Table "):].strip()
            title = _recent_title(body)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            unit = None
            if i < len(lines) and lines[i].startswith("_") and lines[i].rstrip().endswith("_"):
                note = lines[i].strip().strip("_")  # "(단위: 백만원)"
                unit = note.replace("(단위:", "").replace(")", "").strip() or None
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].startswith("|"):
                headers = _split_row(lines[i])
                i += 1
                if i < len(lines) and _is_separator(lines[i]):
                    i += 1
                rows: list[list[str]] = []
                while i < len(lines) and lines[i].startswith("|"):
                    rows.append(_split_row(lines[i]))
                    i += 1
                tables.append(
                    MdTable(company, year, page, table_id, title, unit, headers, rows, file)
                )
            continue
        if line.strip():
            body.append(line.strip())
        i += 1
    return tables


def load_paragraphs(path: Path, min_chars: int = 200) -> list[MdParagraph]:
    """Contiguous body-text blocks (narrative paragraphs), excluding tables."""
    company, year = _company_year(path.stem)
    file = _nfc(path.name)
    lines = _nfc(path.read_text(encoding="utf-8")).splitlines()
    paragraphs: list[MdParagraph] = []
    page = 0
    buf: list[str] = []

    def flush() -> None:
        text = " ".join(buf).strip()
        if len(text) >= min_chars:
            paragraphs.append(MdParagraph(company, year, page, text, file))
        buf.clear()

    for line in lines:
        if line.startswith("## p."):
            flush()
            page = int(line[5:].strip().rstrip("."))
        elif line.startswith("### Table ") or line.startswith("|") or line.startswith("#"):
            flush()
        elif line.startswith("_") and line.rstrip().endswith("_"):
            continue  # unit note
        elif line.strip():
            buf.append(line.strip())
        else:
            flush()
    flush()
    return paragraphs
