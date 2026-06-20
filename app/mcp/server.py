"""MCP server exposing the RAG regression gate as tools (M1/M2).

THIN fastmcp ADAPTER: the projection/statistics core lives in app/core/service.py
(framework-neutral) and is shared with the CLI and the REST API. This file only
registers the core functions as MCP tools — fastmcp is imported ONLY here (the
optional [mcp] extra). Every number comes from the engine via the shared core, so
the MCP path matches scripts/run_gate.py and the REST API exactly.
"""

from __future__ import annotations

from fastmcp import FastMCP

from app.core.analyze import FailureAnalysis, build_analysis
from app.core.service import GateResult, MetricDelta, build_gate_result  # noqa: F401

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
