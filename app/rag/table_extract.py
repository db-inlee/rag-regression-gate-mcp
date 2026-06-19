"""Extract tables and body text from text-based PDFs into Markdown.

Scope (Phase 1.1): text-extractable PDFs only. Scanned / image-only PDFs and OCR
are out of scope. Callers must stop when a PDF is not text-extractable
(`NonTextPDFError`) rather than attempting OCR.

The table normalization logic (merged-cell expansion, multi-level header
flattening) is implemented here as pure functions over the raw cell grid that
pdfplumber returns; it is independent of the extraction library and unit-tested.

Heuristics (documented because they are best-effort, not exact):
  * Header region = leading consecutive rows whose non-empty cells are all
    non-numeric. Tables are assumed to have at least one header row.
  * Horizontally merged header cells are forward-filled from the left.
  * Vertically merged row labels (first column) are forward-filled downward.
  * Unit captions like "(단위: 백만원)" are detected per page and attached above
    each table on that page (positional table<->caption matching is not attempted).
"""

from __future__ import annotations

import re
from pathlib import Path

# Row as returned by pdfplumber: cells may be None for empty / merged cells.
Row = list[str | None]

# Unit caption: "(단위 : 백만원)", "(단위 : 억원, %)" — capture everything up to the
# closing paren / newline so multi-unit captions ("억원, %") are not truncated.
UNIT_CAPTION_RE = re.compile(r"단위\s*[:：]\s*([^)\n]+)")

# Period -> fiscal year mapping found around tables:
#   "제 55 기 2023.01.01 부터 ..."  /  "제 55 기말 2023.12.31 현재"
PERIOD_YEAR_RE = re.compile(r"제\s*(\d+)\s*기말?\s*(\d{4})\.\d{1,2}\.\d{1,2}\s*(?:부터|현재)")

# A header cell that is a bare fiscal period (to be annotated with its year):
#   "제 55 기", "제55기말"
_PERIOD_CELL_RE = re.compile(r"^(제\s*\d+\s*기)(말)?$")

# Looks like a numeric data value: 1,234 / -12.3 / (1,234) / 45% etc.
_NUMERIC_RE = re.compile(r"^[(\-]?[\d,]+(?:\.\d+)?\)?%?$")

# Period/column labels that are numeric but belong in the HEADER, not the data
# body: years (2024, 2024년), Korean fiscal periods (제56기), quarters/halves.
_PERIOD_RE = re.compile(
    r"^(?:(?:19|20)\d{2}년?|제?\s*\d+\s*기|[1-4]\s*분기|상반기|하반기|"
    r"FY\s*\d{2,4}|[1-4]Q|Q[1-4]|\d{4}\.\d{1,2})$"
)

# A text PDF is expected to yield at least this many non-space characters total.
_MIN_TEXT_CHARS = 50


class NonTextPDFError(RuntimeError):
    """Raised when a PDF yields (almost) no extractable text — likely scanned/image."""


# --------------------------------------------------------------------------- #
# Pure table-normalization logic (no I/O, unit-tested)
# --------------------------------------------------------------------------- #

def _cell(value: str | None) -> str:
    return (value or "").strip()


def _looks_numeric(text: str) -> bool:
    return bool(_NUMERIC_RE.match(text.replace(" ", "")))


def _is_period_label(text: str) -> bool:
    return bool(_PERIOD_RE.match(text.replace(" ", "")))


def _is_data_cell(text: str) -> bool:
    """A numeric value that is a data point, not a period/column label."""
    return _looks_numeric(text) and not _is_period_label(text)


def _is_header_row(row: Row) -> bool:
    """A header row has cells but none that look like a data value.

    Year/quarter labels (e.g. '2024') count as header labels, not data — otherwise
    the header region would be cut short on period columns.

    A row with only the first (label) column filled is a section/row label
    (e.g. '자산' on a balance sheet), not a column header — it must not be merged
    into the period header, so it is excluded here and falls through to the body.
    """
    cells = [_cell(c) for c in row]
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    if any(_is_data_cell(c) for c in non_empty):
        return False
    return any(cells[i] for i in range(1, len(cells)))


def _split_header_body(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Leading consecutive header rows vs. the rest (at least one header row)."""
    header_count = 0
    for row in rows:
        if _is_header_row(row):
            header_count += 1
        else:
            break
    header_count = max(header_count, 1)
    return rows[:header_count], rows[header_count:]


def _fill_horizontal(row: Row) -> list[str]:
    """Forward-fill empty cells from the left — covers horizontally merged headers."""
    out: list[str] = []
    last = ""
    for cell in row:
        value = _cell(cell)
        if value:
            last = value
        out.append(value or last)
    return out


def _annotate_period(cell: str, period_year: dict[str, int]) -> str:
    """Append the fiscal year to a bare period label: '제 55 기' -> '제 55 기(2023)'."""
    match = _PERIOD_CELL_RE.match(cell.strip())
    if not match:
        return cell
    key = re.sub(r"\s+", "", match.group(1))  # "제 55 기" -> "제55기"
    year = period_year.get(key)
    return f"{cell}({year})" if year is not None else cell


def _flatten_headers(header_rows: list[Row], period_year: dict[str, int]) -> list[str]:
    """Flatten multi-level headers into one row, joining stacked labels with ' / '.

    Bare fiscal-period labels are annotated with their year so the table is
    self-contained (the year is otherwise only in text outside the table).
    """
    filled = [_fill_horizontal(r) for r in header_rows]
    width = max((len(r) for r in filled), default=0)
    columns: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for r in filled:
            value = r[col] if col < len(r) else ""
            if value:
                value = _annotate_period(value, period_year)
            if value and value not in parts:
                parts.append(value)
        columns.append(" / ".join(parts))
    return columns


def collect_period_years(text: str, period_year: dict[str, int]) -> None:
    """Accumulate period->year mappings from text (mutates period_year in place)."""
    for match in PERIOD_YEAR_RE.finditer(text or ""):
        period_year.setdefault(f"제{match.group(1)}기", int(match.group(2)))


def _fill_label_column(body_rows: list[Row]) -> list[list[str]]:
    """Forward-fill the first column downward — covers vertically merged row labels."""
    out: list[list[str]] = []
    last_label = ""
    for row in body_rows:
        cells = [_cell(c) for c in row]
        if cells:
            if cells[0]:
                last_label = cells[0]
            else:
                cells[0] = last_label
        out.append(cells)
    return out


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _pad(cells: list[str], width: int) -> list[str]:
    return cells + [""] * (width - len(cells))


def table_to_markdown(
    rows: list[Row],
    unit_note: str | None = None,
    period_year: dict[str, int] | None = None,
) -> str:
    """Convert a raw cell grid into a Markdown table, preserving merges/headers/units."""
    rows = [r for r in rows if any(_cell(c) for c in r)]
    if not rows:
        return ""

    header_rows, body_rows = _split_header_body(rows)
    headers = _flatten_headers(header_rows, period_year or {})
    body = _fill_label_column(body_rows)

    width = max([len(headers)] + [len(r) for r in body], default=0)
    headers = _pad(headers, width)

    lines: list[str] = []
    if unit_note:
        lines.append(f"_{unit_note}_")
        lines.append("")
    lines.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in body:
        padded = _pad(row, width)
        lines.append("| " + " | ".join(_md_escape(c) for c in padded) + " |")
    return "\n".join(lines)


def find_unit_captions(page_text: str) -> list[str]:
    """Return unit captions (e.g. '(단위: 백만원)') found in a page's text."""
    captions: list[str] = []
    for match in UNIT_CAPTION_RE.finditer(page_text or ""):
        caption = f"(단위: {match.group(1).strip()})"
        if caption not in captions:
            captions.append(caption)
    return captions


# --------------------------------------------------------------------------- #
# PDF I/O (pdfplumber imported lazily so pure logic / tests need no heavy dep)
# --------------------------------------------------------------------------- #

def is_text_extractable(pdf_path: Path, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """True if the PDF yields enough text to be treated as a text (non-scanned) PDF."""
    import pdfplumber

    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            total += len((page.extract_text() or "").strip())
            if total >= min_chars:
                return True
    return False


def _text_outside_tables(page, table_bboxes: list[tuple]) -> str:
    """Page text with table regions removed, so table content is not duplicated."""
    if not table_bboxes:
        return (page.extract_text() or "").strip()

    def keep(obj) -> bool:
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(
            x0 <= cx <= x1 and top <= cy <= bottom
            for (x0, top, x1, bottom) in table_bboxes
        )

    return (page.filter(keep).extract_text() or "").strip()


def extract_pdf_to_markdown(pdf_path: Path) -> str:
    """Extract a text PDF into Markdown (narrative text + structured tables).

    Table content is emitted only as Markdown tables, not duplicated as raw text.
    Fiscal-period column labels are annotated with their year (e.g. '제55기(2023)').

    Raises NonTextPDFError if the PDF is not text-extractable (scanned/image PDF).
    """
    import pdfplumber

    if not is_text_extractable(pdf_path):
        raise NonTextPDFError(
            f"No extractable text in '{pdf_path}'. Likely a scanned/image PDF; "
            "OCR is out of scope (Phase 1.1). Aborting — provide a text PDF."
        )

    parts: list[str] = [f"# {pdf_path.stem}", ""]
    with pdfplumber.open(pdf_path) as pdf:
        pages = list(pdf.pages)

        # Pass 1: build a document-wide period->year map. Income-statement pages
        # carry "부터", balance-sheet pages carry "현재"; a single doc-wide map lets
        # every table reuse the year even if its own page lacks the mapping line.
        period_year: dict[str, int] = {}
        page_texts: list[str] = []
        for page in pages:
            text = page.extract_text() or ""
            page_texts.append(text)
            collect_period_years(text, period_year)

        # Pass 2: emit narrative text (table regions removed) + structured tables.
        for page_no, page in enumerate(pages, start=1):
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            body_text = _text_outside_tables(page, bboxes)
            units = find_unit_captions(page_texts[page_no - 1])
            if not body_text and not tables:
                continue

            parts.append(f"## p.{page_no}")
            parts.append("")
            if body_text:
                parts.append(body_text)
                parts.append("")

            for table_no, table in enumerate(tables, start=1):
                markdown = table_to_markdown(
                    table.extract(), units[0] if units else None, period_year
                )
                if not markdown:
                    continue
                parts.append(f"### Table p{page_no}-{table_no}")
                parts.append(markdown)
                parts.append("")

    return "\n".join(parts).rstrip() + "\n"
