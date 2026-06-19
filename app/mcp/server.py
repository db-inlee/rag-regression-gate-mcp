"""MCP server exposing the RAG regression gate as a tool (M1).

THIN WRAPPER: this layer reimplements no statistics. It calls the existing engine
(detect.detect_paths -> gate.evaluate -> gate.exit_code) and projects the result
onto Pydantic output models. Every number (delta, CI, significance, verdict, exit
code) comes straight from the engine, so the MCP path matches `scripts/run_gate.py`.

Design (phase-mcp §0): thin wrapper · input = plain str dir paths (same contract as
the run_gate CLI) · output = Pydantic (FastMCP structured content; plain lists, no
self-referential types) · suggestion-only (M1 gives verdict+diagnosis; suggestions
are filled in M2). fastmcp is imported ONLY here and is an optional extra ([mcp]).
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from app.mcp.analyze import FailureAnalysis, build_analysis
from app.mcp.suggest import build_suggestions
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
    """Verdict + failure-mode diagnosis. suggestions filled by M2 (suggestion-only)."""

    verdict: str = Field(description="PASS / WARN / FAIL")
    exit_code: int = Field(description="FAIL=1, WARN/PASS=0 (CI-usable)")
    regressions: list[MetricDelta] = Field(description="significant regressions (FAIL reasons)")
    warnings: list[MetricDelta] = Field(description="borderline signals (one condition only)")
    diagnosis: list[str] = Field(description="human-readable, failure-mode-centric lines")
    suggestions: list[str] = Field(default_factory=list, description="remediation (M2); suggestion-only")


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
        suggestions=build_suggestions(report, baseline_dir, candidate_dir),  # M2 (rule-based)
    )


mcp = FastMCP("rag-regression-gate")


def run_gate(baseline_dir: str, candidate_dir: str) -> GateResult:
    """Compare a candidate RAG run against a baseline and detect quality regressions
    by failure mode (retrieval_miss, hallucination, over_answer). Returns PASS/WARN/FAIL
    with bootstrap CIs and a failure-mode diagnosis. Inputs are directory paths, each
    containing run.jsonl + attribution.jsonl (+ noise_band.json on the baseline).
    Suggestion-only: never modifies configs or runs anything."""
    return build_gate_result(baseline_dir, candidate_dir)


mcp.tool(run_gate)  # register without shadowing the callable (keeps run_gate importable/testable)


def analyze_failures(run_dir: str) -> FailureAnalysis:
    """Diagnose a SINGLE RAG run-log: where is it weak, and what to fix first?
    Pair to run_gate (which compares two runs to detect regressions) — this needs
    only one run_dir (containing run.jsonl + attribution.jsonl). Returns the failure
    distribution, the bottleneck pipeline stage, per-slice failure rates, a
    groundedness breakdown, RAGAS-equivalent metrics (deterministic, no LLM judge),
    and improvement priorities. Gold-free and deterministic (aggregates the
    precomputed attribution; no judge re-run). Suggestion-only: priorities are review
    candidates — apply them, then re-verify with run_gate (closed loop)."""
    return build_analysis(run_dir)


mcp.tool(analyze_failures)  # register without shadowing (keeps it importable/testable)


if __name__ == "__main__":
    mcp.run()
