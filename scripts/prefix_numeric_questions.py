"""One-off: prefix company + report year into numeric question texts (1.2 fix).

table_value and numeric_reasoning-single questions named no company, so the
question text alone was ambiguous across the 3 companies. Prefix
"{company} {fiscal_year}년 보고서 기준, " (company taken from each case).

Comparison questions already name both companies -> left unchanged.
Only `question` text is modified; expected_answer / value / source_ref / gold are
untouched. Idempotent (won't double-prefix). Updates both eval_cases.jsonl and
eval_draft.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

FILES = [Path("data/eval_cases.jsonl"), Path("data/eval_draft.jsonl")]


def needs_prefix(case: dict) -> bool:
    if case["slice"] == "table_value":
        return True
    # numeric_reasoning single (growth) is answer_schema "numeric"; comparison is
    # "comparison" and already names both companies.
    return case["slice"] == "numeric_reasoning" and case["answer_schema"] == "numeric"


def companies_in_question(case: dict) -> bool:
    """Every company named in the case appears in the question text."""
    return all(part and part in case["question"] for part in case["company"].split("|"))


def main() -> int:
    for path in FILES:
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        changed = 0
        for c in rows:
            if needs_prefix(c):
                prefix = f"{c['company']} {c['fiscal_year']}년 보고서 기준, "
                if not c["question"].startswith(prefix):
                    c["question"] = prefix + c["question"]
                    changed += 1
        path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in rows) + "\n", encoding="utf-8")

        numeric = [c for c in rows if c["slice"] in ("table_value", "numeric_reasoning")]
        missing = [c["id"] for c in numeric if not companies_in_question(c)]
        print(f"{path}: prefixed {changed}, numeric cases {len(numeric)}, "
              f"회사명 누락 {len(missing)} {missing[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
