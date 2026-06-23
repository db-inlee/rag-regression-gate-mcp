"""T2 — interface equivalence (pins the multi-way agreement as a test).

The gate's numbers must be identical whichever interface produces them. This pins
scripts/verify_api_equivalence.py's core check as pytest:
  build_gate_result (core)  ==  REST API (/evaluate, in-process TestClient)
  detect_paths (low-level)  ==  build_gate_result (wrapper) — wrapper doesn't distort.

Also confirms the API path does NOT import fastmcp (the A-phase refactor moved the
core into app/core, so app/api depends on app/core only — not app/mcp).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.service import build_gate_result
from app.regression.detect import detect_paths
from app.regression.gate import evaluate, exit_code

BASELINE = "examples/allganize_baseline"
CANDIDATE = "examples/allganize_candidate"

client = TestClient(app)


def test_cli_mcp_api_equivalence():
    """Core (build_gate_result, same fn MCP/CLI call) == REST API /evaluate, exactly."""
    core = build_gate_result(BASELINE, CANDIDATE).model_dump()
    resp = client.post("/evaluate", json={"baseline_dir": BASELINE, "candidate_dir": CANDIDATE})
    assert resp.status_code == 200
    api = resp.json()

    # headline fields
    assert api["verdict"] == core["verdict"]
    assert api["exit_code"] == core["exit_code"]
    assert api["regressions"] == core["regressions"]
    assert api["warnings"] == core["warnings"]
    # full deep-equal (diagnosis + suggestions too) — API adds nothing, distorts nothing
    assert api == core


def test_detect_matches_build_gate_result():
    """The wrapper's verdict/exit_code match the low-level engine (detect_paths→evaluate)."""
    report = detect_paths(Path(BASELINE), Path(CANDIDATE))
    gate = evaluate(report)
    wrapper = build_gate_result(BASELINE, CANDIDATE)

    assert wrapper.verdict == gate["status"]
    assert wrapper.exit_code == exit_code(gate)
    # every significant-regression metric the engine flagged is in the wrapper output
    engine_fail_metrics = {r["metric"] for r in gate["fails"]}
    wrapper_reg_metrics = {r.metric for r in wrapper.regressions}
    assert wrapper_reg_metrics == engine_fail_metrics


def test_api_does_not_import_fastmcp():
    """REST API path is fastmcp-free (refactor: app/api → app/core only, not app/mcp).

    Isolated in a subprocess so the assertion reflects the API's own import graph,
    not whatever other tests in this process may have imported."""
    code = (
        "import app.api.main, sys; "
        "print('fastmcp' in sys.modules or any(m.startswith('app.mcp') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", f"API import pulled fastmcp/app.mcp: {out.stdout!r}"
