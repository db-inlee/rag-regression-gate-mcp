"""Request/response models for the REST API (phase A1).

Response models are REUSED verbatim from the framework-neutral core (app/core:
GateResult / FailureAnalysis) so the API returns the exact same structured output
as the MCP/CLI — zero reimplementation, identical numbers by construction. Only the
small request envelopes are new here. The API imports only app/core (no app/mcp),
so it does NOT depend on fastmcp — just fastapi/uvicorn.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Reused response models (same objects the core returns → API == MCP == CLI).
from app.core.analyze import FailureAnalysis  # noqa: F401  (re-exported for the API)
from app.core.service import GateResult, MetricDelta  # noqa: F401


class EvaluateRequest(BaseModel):
    """Compare two run artifacts. Inputs are directory paths — the SAME contract as
    the run_gate CLI/MCP (each dir holds run.jsonl + attribution.jsonl; baseline also
    noise_band.json), so identical input → identical verdict across all interfaces."""

    baseline_dir: str = Field(description="baseline run dir (run.jsonl + attribution.jsonl + noise_band.json)",
                              examples=["examples/allganize_baseline"])
    candidate_dir: str = Field(description="candidate run dir (run.jsonl + attribution.jsonl)",
                               examples=["examples/allganize_candidate"])


class AnalyzeRequest(BaseModel):
    """Diagnose a single run. `run_dir` holds run.jsonl + attribution.jsonl — same
    contract as the analyze_failures MCP tool."""

    run_dir: str = Field(description="run dir to diagnose (run.jsonl + attribution.jsonl)",
                         examples=["examples/allganize_baseline"])


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
