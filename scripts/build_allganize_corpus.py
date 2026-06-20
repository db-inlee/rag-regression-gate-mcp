"""Build the Allganize law/public corpus (phase D1.2): per-page text extraction.

Streams ONLY the parquet row-groups that contain law/public pages (via a cheap
json-only pre-scan) and extracts each page's PDF bytes -> text (+ tables) with
pdfplumber. Writes one record per page to data/allganize_eval/allganize_corpus.jsonl.

The 40 eval questions reference a subset of these 24 documents; indexing ALL
law+public pages means the rest act as distractors (retrieval is a real signal).

Indexing (chunk + bge-m3 embed) is a separate step in app.adapters.allganize
(build_allganize_index), mirroring the wiki adapter. This script does extraction only.

Source : datalama/RAG-Evaluation-Dataset-KO (data/test parquet, pdf column)
License: MIT (see data/allganize_eval/LICENSE.md)

Usage: python scripts/build_allganize_corpus.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pdfplumber
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

PARQUET = "datasets/datalama/RAG-Evaluation-Dataset-KO/data/test-00000-of-00001.parquet"
DOMAINS = {"law", "public"}
OUT = Path("data/allganize_eval/allganize_corpus.jsonl")


def _page_text(pdf_bytes: bytes) -> str:
    """Extract page text + tables (pipe-rendered) so table cell values are indexed."""
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:  # each record is a single-page PDF
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt.strip())
            for table in page.extract_tables() or []:
                rows = [" | ".join((c or "").strip() for c in row) for row in table if any(row)]
                if rows:
                    parts.append("\n".join(rows))
    return "\n\n".join(parts).strip()


def main() -> int:
    fs = HfFileSystem()
    f = fs.open(PARQUET, "rb")
    pf = pq.ParquetFile(f)

    # pass 1 (cheap, json only): which row-groups hold law/public rows?
    need = []
    for rg in range(pf.num_row_groups):
        doms = {r["domain"] for r in pf.read_row_group(rg, columns=["json"]).to_pydict()["json"]}
        if doms & DOMAINS:
            need.append(rg)
    print(f"row-groups with law/public: {need} ({len(need)}/{pf.num_row_groups})")

    # pass 2: read pdf bytes only for those row-groups; extract page text
    rows: list[dict] = []
    empty_pages = 0
    for rg in need:
        tbl = pf.read_row_group(rg, columns=["json", "pdf"]).to_pydict()
        for meta, pdf_bytes in zip(tbl["json"], tbl["pdf"]):
            if meta["domain"] not in DOMAINS:
                continue
            text = _page_text(pdf_bytes)
            if not text:
                empty_pages += 1
            rows.append({
                "pid": meta["pid"],
                "file_name": meta["file_name"],
                "domain": meta["domain"],
                "page_number": meta["page_number"],
                "text": text,
            })
        print(f"  row-group {rg}: cumulative {len(rows)} pages")

    rows.sort(key=lambda r: (int(r["pid"].split("_")[1]), r["page_number"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    docs = {r["pid"] for r in rows}
    by_domain = {d: len({r["pid"] for r in rows if r["domain"] == d}) for d in sorted(DOMAINS)}
    chars = sum(len(r["text"]) for r in rows)
    print(f"\nwrote {len(rows)} pages from {len(docs)} documents -> {OUT}")
    print(f"  documents per domain : {by_domain}")
    print(f"  pages with empty text: {empty_pages}/{len(rows)} (likely image-only)")
    print(f"  total text chars     : {chars:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
