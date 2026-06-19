"""Run the regression gate (Phase 3.3 / 6.1): detect → PASS/WARN/FAIL → exit code.

CI mode (gold-free, no LLM/embedding — only enriched attribution + noise_band):
  python scripts/run_gate.py --baseline examples/baseline --candidate examples/candidate

Local/back-compat mode (resolves run_ids in reports/, recomputes from gold):
  python scripts/run_gate.py <baseline_run_id> <candidate_run_id>

exit code: FAIL=1, WARN=0, PASS=0 (CI-usable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.regression.detect import detect, detect_paths
from app.regression.gate import evaluate, exit_code, render


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="baseline dir (run.jsonl + attribution.jsonl + noise_band.json)")
    ap.add_argument("--candidate", help="candidate dir (run.jsonl + attribution.jsonl)")
    ap.add_argument("baseline_run", nargs="?", help="(local) baseline run_id in reports/")
    ap.add_argument("candidate_run", nargs="?", help="(local) candidate run_id in reports/")
    args = ap.parse_args()

    if args.baseline and args.candidate:
        report = detect_paths(Path(args.baseline), Path(args.candidate))
        cand_name = Path(args.candidate).name
    elif args.baseline_run and args.candidate_run:
        report = detect(args.baseline_run, args.candidate_run)
        cand_name = args.candidate_run
    else:
        print("usage: run_gate.py --baseline <dir> --candidate <dir>"
              "  |  run_gate.py <baseline_run_id> <candidate_run_id>")
        return 2

    gate = evaluate(report)
    text = render(report, gate)
    print(text)

    # gate outputs are regenerable byproducts → kept OUT of reports/ (originals only)
    out = Path("gate_runs") / f"gate_{cand_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gate": gate["status"], **report, "gate_detail": gate},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")  # CI visibility
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## RAG Regression Gate\n\n```\n" + text + "\n```\n")

    code = exit_code(gate)
    print(f"\nexit code: {code}  ({gate['status']})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
