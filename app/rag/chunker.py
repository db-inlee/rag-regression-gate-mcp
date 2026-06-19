"""Table-aware chunking of extracted corpus Markdown (Phase 1.3.1).

Tables (`### Table` blocks) are kept whole when they fit in chunk_size. When a
table is larger than chunk_size it is split by data rows, and the header row
(with year columns) + separator + unit caption are **repeated on every fragment**
so each fragment is self-describing for retrieval. Non-table prose is split with
LangChain's RecursiveCharacterTextSplitter.

Chunk metadata: {company, fiscal_year, source_file, is_table, section, page, table_id}.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RagConfig

# A prose line that looks like a section heading (best-effort metadata).
_HEADING_RE = re.compile(r"^\s*(?:[0-9]+(?:[-.][0-9]+)*\.|[IVXⅠ-Ⅻ]+\.|[가-힣]\.)\s*\S")
_SECTION_KW = ("손익계산서", "재무상태표", "현금흐름표", "포괄손익", "사업의 개요",
               "사업의 내용", "위험", "연구개발", "주요 계약", "주주", "임원")


@dataclass
class Chunk:
    text: str
    metadata: dict


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _company_year(stem: str) -> tuple[str, int]:
    company, _, year = _nfc(stem).rpartition("_")
    return company, int(year)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return set(line.strip()) <= {"|", "-", " "} and "-" in line


def _looks_like_heading(line: str) -> bool:
    return len(line) <= 40 and (bool(_HEADING_RE.match(line)) or any(k in line for k in _SECTION_KW))


def _render_table(headers: list[str], rows: list[list[str]], unit: str | None) -> str:
    lines: list[str] = []
    if unit:
        lines.append(f"(단위: {unit})")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _table_fragments(headers: list[str], rows: list[list[str]], unit: str | None,
                     chunk_size: int) -> list[str]:
    """One string if it fits; else row-grouped fragments, header repeated on each."""
    full = _render_table(headers, rows, unit)
    if len(full) <= chunk_size or not rows:
        return [full]
    fragments: list[str] = []
    group: list[list[str]] = []
    for row in rows:
        if group and len(_render_table(headers, group + [row], unit)) > chunk_size:
            fragments.append(_render_table(headers, group, unit))
            group = [row]
        else:
            group.append(row)
    if group:
        fragments.append(_render_table(headers, group, unit))
    return fragments


def chunk_file(path: Path, config: RagConfig) -> list[Chunk]:
    company, year = _company_year(path.stem)
    source_file = _nfc(path.name)
    lines = _nfc(path.read_text(encoding="utf-8")).splitlines()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )

    chunks: list[Chunk] = []
    page = 0
    section = ""
    prose: list[str] = []

    def meta(is_table: bool, table_id: str = "") -> dict:
        return {
            "company": company,
            "fiscal_year": year,
            "source_file": source_file,
            "is_table": is_table,
            "section": section or f"p.{page}",
            "page": page,
            "table_id": table_id,
        }

    def flush_prose() -> None:
        text = "\n".join(prose).strip()
        prose.clear()
        if not text:
            return
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(Chunk(piece.strip(), meta(is_table=False)))

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## p."):
            flush_prose()
            page = int(line[5:].strip().rstrip("."))
            i += 1
            continue
        if line.startswith("### Table "):
            flush_prose()
            table_id = line[len("### Table "):].strip()
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            unit = None
            if i < len(lines) and lines[i].startswith("_") and lines[i].rstrip().endswith("_"):
                unit = lines[i].strip().strip("_").replace("(단위:", "").replace(")", "").strip() or None
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            headers: list[str] = []
            rows: list[list[str]] = []
            if i < len(lines) and lines[i].startswith("|"):
                headers = _split_row(lines[i])
                i += 1
                if i < len(lines) and _is_separator(lines[i]):
                    i += 1
                while i < len(lines) and lines[i].startswith("|"):
                    rows.append(_split_row(lines[i]))
                    i += 1
            for frag in _table_fragments(headers, rows, unit, config.chunk_size):
                chunks.append(Chunk(frag, meta(is_table=True, table_id=table_id)))
            continue
        if line.startswith("#"):
            i += 1
            continue
        if line.strip():
            if _looks_like_heading(line.strip()):
                section = line.strip()
            prose.append(line.strip())
        else:
            flush_prose()
        i += 1
    flush_prose()
    return chunks


def chunk_corpus(extracted_dir: Path, config: RagConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(extracted_dir.glob("*.md")):
        chunks.extend(chunk_file(path, config))
    return chunks
