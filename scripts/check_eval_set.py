"""Phase 1.2b gate — validate the final eval set (PASS/FAIL, exit code).

Checks (per tasks/phase-1.2.md Done + CLAUDE.md):
  * exactly 100 lines, all valid EvalCase
  * slice distribution 40 / 20 / 25 / 15
  * numeric_reasoning has 8 comparison cases (each with >=2 companies + comparison)
  * no_answer: 15, unanswerable, empty contexts, sentinel answer
  * gold_failure_type == "correct" for every case
  * slice <-> answer_schema consistency
  * unique ids; source_ref present and NOT a forbidden (wide-spec) table

Usage:
  python scripts/check_eval_set.py                      # data/eval_cases.jsonl
  python scripts/check_eval_set.py data/eval_draft.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import EvalCase

DEFAULT_PATH = Path("data/eval_cases.jsonl")
EXPECTED_SLICES = {"table_value": 40, "numeric_reasoning": 20, "body_text": 25, "no_answer": 15}
SLICE_SCHEMAS = {
    "table_value": {"numeric"},
    "numeric_reasoning": {"numeric", "comparison"},
    "body_text": {"text"},
    "no_answer": {"no_answer"},
}
# Wide spec tables with broken cell extraction (see KNOWN_ISSUES.md) — never a source.
FORBIDDEN_REF = ("명세", "종속기업", "관계기업", "출자", "타법인")


def _load(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_checks(cases: list[dict]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    # 1. line count
    check("100 lines", len(cases) == 100, f"got {len(cases)}")

    # 2. schema validity (covers expected_answer <-> answer_schema)
    invalid = []
    for c in cases:
        try:
            EvalCase.model_validate(c)
        except Exception as exc:  # noqa: BLE001
            invalid.append(f"{c.get('id', '?')}: {exc}")
    check("EvalCase schema", not invalid, "; ".join(invalid[:3]))

    # 3. slice distribution
    counts = Counter(c.get("slice") for c in cases)
    bad = {s: counts.get(s, 0) for s, n in EXPECTED_SLICES.items() if counts.get(s, 0) != n}
    check("slice 40/20/25/15", not bad, f"off: {bad}" if bad else "")

    # 4. comparison cases
    comp = [c for c in cases if c.get("answer_schema") == "comparison"]
    bad_comp = [c["id"] for c in comp
                if len((c.get("expected_answer") or {}).get("companies", {})) < 2
                or not (c.get("expected_answer") or {}).get("comparison")]
    check("comparison == 8 (>=2 companies + comparison)",
          len(comp) == 8 and not bad_comp,
          f"count={len(comp)} bad={bad_comp}")

    # 5. no_answer integrity
    na = [c for c in cases if c.get("slice") == "no_answer"]
    bad_na = [c["id"] for c in na
              if c.get("answer_type") != "unanswerable"
              or c.get("contexts")
              or (c.get("expected_answer") or {}).get("sentinel") != "정보 없음"]
    check("no_answer == 15 (unanswerable, empty contexts, sentinel)",
          len(na) == 15 and not bad_na, f"count={len(na)} bad={bad_na}")

    # 6. gold_failure_type all correct
    bad_gold = [c.get("id") for c in cases if c.get("gold_failure_type") != "correct"]
    check("gold_failure_type all 'correct'", not bad_gold, str(bad_gold[:5]))

    # 7. slice <-> answer_schema consistency
    bad_schema = [c.get("id") for c in cases
                  if c.get("answer_schema") not in SLICE_SCHEMAS.get(c.get("slice"), set())]
    check("slice <-> answer_schema", not bad_schema, str(bad_schema[:5]))

    # 8. unique ids
    ids = [c.get("id") for c in cases]
    dups = [i for i, n in Counter(ids).items() if n > 1]
    check("unique ids", not dups, str(dups))

    # 9. source_ref present and not forbidden
    no_ref = [c.get("id") for c in cases if not c.get("source_ref")]
    forbidden = [c.get("id") for c in cases
                 if any(k in (c.get("source_ref") or "") for k in FORBIDDEN_REF)]
    check("source_ref valid & not forbidden table",
          not no_ref and not forbidden, f"missing={no_ref[:3]} forbidden={forbidden[:3]}")

    return results


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    cases = _load(path)
    if cases is None:
        print(f"FAIL: {path} not found. Run scripts/review_eval_set.py to produce it.")
        return 1

    print(f"checking {path} ({len(cases)} cases)\n")
    results = run_checks(cases)
    for name, ok, detail in results:
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
        if not ok and detail:
            line += f"  — {detail}"
        print(line)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    ok = passed == total
    print(f"\n{'PASS' if ok else 'FAIL'}: {passed}/{total} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
