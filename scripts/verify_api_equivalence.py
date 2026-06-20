"""A1 verification — the REST API returns the SAME numbers as CLI and MCP.

Same input (allganize baseline vs candidate):
  API  /evaluate  (FastAPI TestClient)
  ==  MCP  build_gate_result
  ==  CLI  scripts/run_gate.py  (verdict + exit code + per-metric rows)
and  API /analyze == MCP build_analysis.

Proves the API adds no logic and doesn't distort the core (4th agreement point).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.api.main import app
from app.core.analyze import build_analysis
from app.core.service import build_gate_result

BASE = "examples/allganize_baseline"
CAND = "examples/allganize_candidate"
KEYS = ("metric", "delta", "ci_low", "ci_high", "significant", "direction")


def _rows(items: list[dict]) -> set:
    """Comparable signature for a list of metric-delta rows (order-independent)."""
    return {tuple(round(r[k], 6) if isinstance(r[k], float) else r[k] for k in KEYS) for r in items}


def main() -> int:
    client = TestClient(app)
    ok = True

    # --- smoke: /health and / redirect ---
    h = client.get("/health")
    print(f"GET /health  -> {h.status_code}  {h.json()}")
    r = client.get("/", follow_redirects=False)
    print(f"GET /        -> {r.status_code}  redirect to {r.headers.get('location')}")
    ok &= h.status_code == 200 and r.status_code in (307, 308)

    # --- MCP core ---
    mcp = build_gate_result(BASE, CAND).model_dump()

    # --- API /evaluate ---
    api = client.post("/evaluate", json={"baseline_dir": BASE, "candidate_dir": CAND})
    assert api.status_code == 200, api.text
    api = api.json()

    # --- CLI ---
    proc = subprocess.run([sys.executable, "scripts/run_gate.py", "--baseline", BASE, "--candidate", CAND],
                          capture_output=True, text=True)
    cli_json = json.loads(Path(f"gate_runs/gate_{Path(CAND).name}.json").read_text(encoding="utf-8"))
    cli_verdict = cli_json["gate"]
    cli_fails = _rows(cli_json["gate_detail"]["fails"])
    cli_warns = _rows(cli_json["gate_detail"]["warns"])
    cli_exit = proc.returncode

    print("\n=== /evaluate equivalence (API == MCP == CLI) ===")
    print(f"{'':14}{'verdict':>8}{'exit':>6}  fails / warns")
    print(f"{'MCP':14}{mcp['verdict']:>8}{mcp['exit_code']:>6}  {len(mcp['regressions'])} / {len(mcp['warnings'])}")
    print(f"{'API':14}{api['verdict']:>8}{api['exit_code']:>6}  {len(api['regressions'])} / {len(api['warnings'])}")
    print(f"{'CLI':14}{cli_verdict:>8}{cli_exit:>6}  {len(cli_json['gate_detail']['fails'])} / {len(cli_json['gate_detail']['warns'])}")

    verdict_ok = mcp["verdict"] == api["verdict"] == cli_verdict
    exit_ok = mcp["exit_code"] == api["exit_code"] == cli_exit
    fails_ok = _rows(mcp["regressions"]) == _rows(api["regressions"]) == cli_fails
    warns_ok = _rows(mcp["warnings"]) == _rows(api["warnings"]) == cli_warns
    print(f"  verdict identical : {verdict_ok}")
    print(f"  exit_code identical: {exit_ok}")
    print(f"  regressions rows identical: {fails_ok}")
    print(f"  warnings rows identical   : {warns_ok}")
    # full deep-equal API vs MCP (every field incl. diagnosis/suggestions)
    api_full_ok = api == mcp
    print(f"  API == MCP (full model deep-equal): {api_full_ok}")
    ok &= verdict_ok and exit_ok and fails_ok and warns_ok and api_full_ok

    # --- /analyze equivalence (API == MCP) ---
    mcp_an = build_analysis(BASE).model_dump()
    api_an = client.post("/analyze", json={"run_dir": BASE})
    assert api_an.status_code == 200, api_an.text
    api_an = api_an.json()
    analyze_ok = api_an == mcp_an
    print("\n=== /analyze equivalence (API == MCP) ===")
    print(f"  bottleneck_stage: MCP={mcp_an['bottleneck_stage']}  API={api_an['bottleneck_stage']}")
    print(f"  API == MCP (full deep-equal): {analyze_ok}")
    ok &= analyze_ok

    print(f"\n{'✅ ALL EQUIVALENT' if ok else '❌ MISMATCH'}  (API == MCP == CLI)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
