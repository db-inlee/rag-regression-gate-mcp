"""Entrypoint: extract tables/text from data/corpus/raw/*.pdf into Markdown.

Input  : data/corpus/raw/{company}_{year}.pdf  (you place PDFs here manually)
Output : data/corpus/extracted/{company}_{year}.md

Filename rule: "{company}_{year}.pdf" with a 4-digit year, e.g. "삼성전자_2024.pdf".
Files that do not match are reported and skipped (not guessed).

If any PDF is not text-extractable (scanned/image), the run stops with an error
(OCR is out of scope, per Phase 1.1).
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# Make `app` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.table_extract import NonTextPDFError, extract_pdf_to_markdown

RAW_DIR = Path("data/corpus/raw")
OUT_DIR = Path("data/corpus/extracted")
FILENAME_RE = re.compile(r"^(?P<company>.+)_(?P<year>\d{4})$")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("extract_tables")


def parse_company_year(pdf_path: Path) -> tuple[str, int] | None:
    match = FILENAME_RE.match(pdf_path.stem)
    if not match:
        return None
    return match.group("company"), int(match.group("year"))


def main() -> int:
    logger.info("input dir : %s", RAW_DIR.resolve())
    logger.info("output dir: %s", OUT_DIR.resolve())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("no PDFs found in %s — place reports there first.", RAW_DIR)
        return 1
    logger.info("found %d PDF(s): %s", len(pdfs), [p.name for p in pdfs])

    bad_names: list[str] = []
    written = 0
    for pdf in pdfs:
        parsed = parse_company_year(pdf)
        if parsed is None:
            bad_names.append(pdf.name)
            logger.warning("skip '%s': name must be '{company}_{year}.pdf'", pdf.name)
            continue
        company, year = parsed

        try:
            markdown = extract_pdf_to_markdown(pdf)
        except NonTextPDFError as exc:
            logger.error("%s", exc)
            return 2  # stop the whole run; do not attempt OCR

        out_path = OUT_DIR / f"{company}_{year}.md"
        out_path.write_text(markdown, encoding="utf-8")
        written += 1
        logger.info("wrote %s (%d chars)", out_path, len(markdown))

    logger.info("done: %d extracted, %d skipped (bad name)", written, len(bad_names))
    if bad_names:
        logger.warning("skipped (rename to {company}_{year}.pdf): %s", bad_names)
        return 3 if written == 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
