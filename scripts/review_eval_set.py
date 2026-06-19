"""Phase 1.2b — interactive review of the eval-set draft.

Shows each case [question / source_ref / contexts snippet / proposed answer] and
records approve (✓) / discard (✗) / edit. Progress is persisted, so you can quit
(q) any time and resume where you left off.

Review order: body_text first (LLM-drafted, needs the most scrutiny), then the
numeric slices for a spot-check, then no_answer. Use --slice to review just one.

Confirmed cases are written to data/eval_cases.jsonl with `needs_review` removed
and gold_failure_type forced to "correct" (validated against EvalCase).

Usage:
  python scripts/review_eval_set.py                # review all, in order
  python scripts/review_eval_set.py --slice numeric_reasoning   # one slice
  python scripts/review_eval_set.py --reset        # clear saved progress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import EvalCase

DRAFT_PATH = Path("data/eval_draft.jsonl")
OUT_PATH = Path("data/eval_cases.jsonl")
STATE_PATH = Path("data/eval_review_state.json")

REVIEW_ORDER = ["body_text", "table_value", "numeric_reasoning", "no_answer"]
SNIPPET_CHARS = 800
RULE = "─" * 78


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def load_draft() -> list[dict]:
    if not DRAFT_PATH.exists():
        sys.exit(f"draft not found: {DRAFT_PATH} (run scripts/build_eval_set.py first)")
    return [json.loads(line) for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"decisions": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_case(draft: dict, decision: dict) -> dict:
    """Apply an approve/edit decision -> a confirmed EvalCase dict."""
    case = dict(draft)
    case.pop("needs_review", None)
    case["gold_failure_type"] = "correct"
    if decision.get("question"):
        case["question"] = decision["question"]
    if decision.get("expected_answer") is not None:
        case["expected_answer"] = decision["expected_answer"]
    return case


def write_eval_cases(draft: list[dict], state: dict) -> tuple[int, int]:
    """Write approved/edited cases to OUT_PATH. Returns (written, invalid)."""
    decisions = state["decisions"]
    written = invalid = 0
    lines: list[str] = []
    for c in draft:
        d = decisions.get(c["id"])
        if not d or d["decision"] == "discard":
            continue
        case = finalize_case(c, d)
        try:
            EvalCase.model_validate(case)
        except Exception as exc:  # noqa: BLE001
            invalid += 1
            print(f"  ! {c['id']} invalid, skipped from output: {exc}")
            continue
        lines.append(json.dumps(case, ensure_ascii=False))
        written += 1
    OUT_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return written, invalid


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #

def ordered_cases(draft: list[dict], slice_filter: str | None) -> list[dict]:
    cases = [c for c in draft if slice_filter is None or c["slice"] == slice_filter]

    def key(c: dict):
        primary = REVIEW_ORDER.index(c["slice"]) if c["slice"] in REVIEW_ORDER else 99
        # group table_value by source table so same-table cases are adjacent
        secondary = c["source_ref"] if c["slice"] == "table_value" else ""
        return (primary, secondary, c["id"])

    return sorted(cases, key=key)


def render(case: dict, idx: int, total: int, prev: dict | None) -> None:
    print("\n" + RULE)
    flag = ""
    if prev:
        flag = f"  (이전 결정: {prev['decision']})"
    print(f"[{idx}/{total}]  {case['id']}  ·  slice={case['slice']}  ·  schema={case['answer_schema']}{flag}")
    print(RULE)
    print(f"질문        : {case['question']}")
    print(f"source_ref  : {case['source_ref']}")
    print(f"answer_type : {case['answer_type']}")
    print("\ncontexts (gold 근거):")
    if not case["contexts"]:
        print("  (없음 — no_answer)")
    for ctx in case["contexts"]:
        snippet = ctx.strip().replace("\n", "\n  ")
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + " …(생략)"
        print("  " + snippet)
    print("\n제안 정답   :")
    print("  " + json.dumps(case["expected_answer"], ensure_ascii=False, indent=2).replace("\n", "\n  "))
    print(RULE)


# --------------------------------------------------------------------------- #
# Edit
# --------------------------------------------------------------------------- #

def prompt_edit(case: dict) -> dict | None:
    """Collect an edited question/expected_answer; return decision dict or None."""
    print("편집 (그냥 Enter = 변경 안 함):")
    new_q = input("  새 질문> ").strip()
    print('  새 정답(JSON 한 줄, 예: {"value": 123, "unit": "백만원", "basis": "연결", "period": "2024(제56기)"})')
    raw = input("  새 정답> ").strip()

    decision: dict = {"decision": "edit"}
    if new_q:
        decision["question"] = new_q
    if raw:
        try:
            decision["expected_answer"] = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  ! JSON 파싱 실패: {exc} — 편집 취소")
            return None

    candidate = finalize_case(case, decision)
    try:
        EvalCase.model_validate(candidate)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! 스키마 검증 실패: {exc} — 편집 취소")
        return None
    if "question" not in decision and "expected_answer" not in decision:
        print("  (변경 없음 — 승인으로 처리)")
        return {"decision": "approve"}
    return decision


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def summarize(draft: list[dict], state: dict) -> None:
    decisions = state["decisions"]
    n = len(draft)
    approve = sum(1 for d in decisions.values() if d["decision"] in ("approve", "edit"))
    discard = sum(1 for d in decisions.values() if d["decision"] == "discard")
    print(f"\n진행 상황: 결정 {len(decisions)}/{n}  (승인/수정 {approve} · 폐기 {discard} · 미결정 {n - len(decisions)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=REVIEW_ORDER, help="review only this slice")
    ap.add_argument("--ids", help="comma-separated case ids to review (e.g. case_044,case_082)")
    ap.add_argument("--reset", action="store_true", help="clear saved progress and exit")
    ap.add_argument("--redo", action="store_true", help="re-review already-decided cases too")
    args = ap.parse_args()

    if args.reset:
        STATE_PATH.unlink(missing_ok=True)
        print(f"cleared {STATE_PATH}")
        return 0

    draft = load_draft()
    state = load_state()
    decisions = state["decisions"]

    cases = ordered_cases(draft, args.slice)

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        missing = wanted - {c["id"] for c in draft}
        if missing:
            print(f"주의: 존재하지 않는 id {sorted(missing)}")
        pending = [c for c in cases if c["id"] in wanted]  # show regardless of prior decisions
        scope = f"ids={len(pending)}개"
    else:
        pending = [c for c in cases if args.redo or c["id"] not in decisions]
        scope = f"slice={args.slice or 'ALL'}"

    print(f"draft: {len(draft)} cases · 이번 검수 대상: {len(pending)} "
          f"({scope}, 순서: {' → '.join(REVIEW_ORDER)})")
    print("명령: [a]승인  [d]폐기  [e]수정  [s]건너뛰기  [b]뒤로  [q]저장후종료")

    i = 0
    while i < len(pending):
        case = pending[i]
        render(case, i + 1, len(pending), decisions.get(case["id"]))
        try:
            choice = input("결정 [a/d/e/s/b/q]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n중단 — 진행 상황 저장.")
            break

        if choice in ("a", "✓", "y"):
            decisions[case["id"]] = {"decision": "approve"}
        elif choice in ("d", "✗", "n"):
            decisions[case["id"]] = {"decision": "discard"}
        elif choice == "e":
            edited = prompt_edit(case)
            if edited is None:
                continue  # re-show same case
            decisions[case["id"]] = edited
        elif choice == "s":
            pass  # leave undecided
        elif choice == "b":
            i = max(0, i - 1)
            continue
        elif choice == "q":
            print("저장 후 종료.")
            break
        else:
            print("  ? a/d/e/s/b/q 중 하나를 입력하세요.")
            continue

        save_state(state)
        i += 1

    save_state(state)
    written, invalid = write_eval_cases(draft, state)
    summarize(draft, state)
    print(f"확정본 기록: {written} cases -> {OUT_PATH}" + (f"  (검증 실패 {invalid}건 제외)" if invalid else ""))
    if written != len(draft):
        print(f"주의: 확정 {written} / draft {len(draft)} — 100개 충족하려면 폐기분 대체/미결정 검수 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
