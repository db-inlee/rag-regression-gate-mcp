"""Framework-neutral service core (phase A refactor).

The interface core — build_gate_result (run_gate) and build_analysis
(analyze_failures) — lives here, depending only on the engine
(app.regression / app.core.analyze / app.core.suggest). NO web framework: the MCP
adapter (app/mcp/server.py, fastmcp) and the REST adapter (app/api, fastapi) both
import from this module, so neither pulls the other's dependency.

This is a pure MOVE of the projection logic out of app/mcp/server.py — same
functions, same numbers. The engine (regression/evaluator) is untouched.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

# build_analysis / FailureAnalysis re-exported so this module is the single core surface.
from app.core.analyze import FailureAnalysis, build_analysis  # noqa: F401
from app.core.suggest import build_suggestions
from app.regression.detect import detect_paths
from app.regression.gate import evaluate, exit_code, line


class MetricDelta(BaseModel):
    """One metric's baseline→candidate change (projected from a detect result row)."""

    metric: str
    baseline: float | None
    candidate: float | None
    delta: float
    ci_low: float
    ci_high: float
    significant: bool
    direction: str  # regression / improvement / warn / no_change


class GateResult(BaseModel):
    """Verdict + failure-mode diagnosis + rule-based suggestions (suggestion-only)."""

    verdict: str = Field(description="PASS / WARN / FAIL")
    exit_code: int = Field(description="FAIL=1, WARN/PASS=0 (CI-usable)")
    regressions: list[MetricDelta] = Field(description="significant regressions (FAIL reasons)")
    warnings: list[MetricDelta] = Field(description="borderline signals (one condition only)")
    diagnosis: list[str] = Field(description="human-readable, failure-mode-centric lines")
    suggestions: list[str] = Field(default_factory=list, description="remediation; suggestion-only")


def _delta(r: dict) -> MetricDelta:
    """Pure projection of a detect result row onto MetricDelta (no logic)."""
    return MetricDelta(
        metric=r["metric"], baseline=r["baseline"], candidate=r["candidate"],
        delta=r["delta"], ci_low=r["ci_low"], ci_high=r["ci_high"],
        significant=r["significant"], direction=r["direction"],
    )


def build_gate_result(baseline_dir: str, candidate_dir: str) -> GateResult:
    """Call the engine and project the result. The ONLY logic here is field mapping."""
    report = detect_paths(Path(baseline_dir), Path(candidate_dir))  # bootstrap CI + band
    gate = evaluate(report)                                         # PASS/WARN/FAIL + buckets
    # diagnosis reuses gate.line() → identical wording to the CLI, zero reimplementation.
    diagnosis = [line(r) for r in gate["fails"]] + [line(r, significant=False) for r in gate["warns"]]
    return GateResult(
        verdict=gate["status"],
        exit_code=exit_code(gate),
        regressions=[_delta(r) for r in gate["fails"]],
        warnings=[_delta(r) for r in gate["warns"]],
        diagnosis=diagnosis,
        suggestions=build_suggestions(report, baseline_dir, candidate_dir),  # rule-based
    )
