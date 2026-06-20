"""REST API for the RAG regression gate (phase A1) — the 3rd interface.

CLI (scripts/run_gate.py) · MCP (app/mcp/) · REST API (here) all call the SAME
framework-neutral core (app/core/service.py) — this layer is a thin FastAPI adapter,
no statistics reimplemented:

  POST /evaluate → app.core.service.build_gate_result   (run_gate core)
  POST /analyze  → app.core.analyze.build_analysis      (analyze_failures core)

So the same input yields the same verdict/numbers across CLI == MCP == API. The API
imports only app/core (not app/mcp) → depends on fastapi/uvicorn, NOT fastmcp.
Engine is untouched (0-diff); only app/api/ + app/core/ are new/moved.

Run: uvicorn app.api.main:app --reload   (deps: requirements-api.txt)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.schemas import (AnalyzeRequest, EvaluateRequest, FailureAnalysis,
                             GateResult, HealthResponse)
from app.core.analyze import build_analysis
from app.core.service import build_gate_result

API_VERSION = "0.1.0"

logger = logging.getLogger("rag_api")

# Real, representative payloads (from examples/allganize_*) shown in /docs & /redoc.
_EVALUATE_EXAMPLE = {
    "verdict": "FAIL",
    "exit_code": 1,
    "regressions": [
        {"metric": "answerable_accuracy", "baseline": 0.55, "candidate": 0.3,
         "delta": -0.25, "ci_low": -0.4, "ci_high": -0.125, "significant": True,
         "direction": "regression"},
        {"metric": "retrieval_miss", "baseline": 1, "candidate": 5, "delta": 4,
         "ci_low": 1, "ci_high": 8, "significant": True, "direction": "regression"},
    ],
    "warnings": [],
    "diagnosis": ["정답 정확도(grounded) 0.5500→0.3000 유의 하락(Δ-0.2500, CI [-0.4000, -0.1250]) → 정확도 회귀"],
    "suggestions": ["[retrieval_miss] 검색 단계 회귀 … 원인 후보: top_k 5→1 → 우선 되돌림(top_k 1→5) 검토 …"],
}
_ANALYZE_EXAMPLE = {
    "n_cases": 40,
    "failure_distribution": {"correct": 22, "hallucination": 17, "retrieval_miss": 1},
    "bottleneck_stage": "grounding",
    "bottleneck_reason": "hallucination가 17건으로 가장 큰 병목 (그라운딩 단계)",
    "grounded_correct": 22,
    "unsupported_correct": 0,
    "ragas_equivalent": {"context_recall": 0.975, "faithfulness": 1.0, "answer_correctness": 0.55},
    "improvement_priorities": ["우선순위 1: [그라운딩 단계 — hallucination 17건] → 근거 인용 강제(citation) …"],
}


def _require_artifacts(run_dir: str, files: tuple[str, ...]) -> None:
    """404 with a precise message if the run dir or a required artifact is missing.

    (Pydantic already returns 422 for missing/mistyped request fields; this guards
    the filesystem contract — clearer than letting FileNotFoundError become a 500.)"""
    d = Path(run_dir)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"run directory not found: {run_dir}")
    missing = [f for f in files if not (d / f).is_file()]
    if missing:
        raise HTTPException(status_code=404,
                            detail=f"missing artifact(s) in {run_dir}: {missing}")

app = FastAPI(
    title="RAG Regression Gate API",
    version=API_VERSION,
    description=(
        "RAG 회귀 게이트를 REST로 노출하는 평가 서비스 — CLI/MCP와 같은 코어를 호출하는 3번째 인터페이스.\n\n"
        "- **POST /evaluate** — 두 run(baseline↔candidate) 비교 → PASS/WARN/FAIL (run_gate 코어)\n"
        "- **POST /analyze** — 단일 run 진단 → 병목·슬라이스·RAGAS·우선순위 (analyze_failures 코어)\n\n"
        "입력은 run 아티팩트(run.jsonl + attribution.jsonl) 디렉토리 경로(CLI/MCP와 동일 계약). "
        "같은 입력 → 같은 수치(결정적). RAG 실행 자체는 범위 밖 — 이 API는 run-log/attribution을 받아 판정한다."
    ),
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Per-request timing + result log (not excessive: one line per request)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                response.status_code, dt)
    return response


@app.exception_handler(json.JSONDecodeError)
async def _malformed_artifact(request: Request, exc: json.JSONDecodeError):
    """A run/attribution file exists but isn't valid JSONL → 422 (client data error)."""
    return JSONResponse(status_code=422, content={"detail": f"malformed artifact (invalid JSONL): {exc}"})


@app.exception_handler(FileNotFoundError)
async def _missing_file(request: Request, exc: FileNotFoundError):
    """Safety net if a required file slips past _require_artifacts → 404."""
    return JSONResponse(status_code=404, content={"detail": f"artifact not found: {exc}"})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(status="ok", version=API_VERSION)


@app.post("/evaluate", response_model=GateResult, tags=["gate"],
          summary="Compare two runs → PASS/WARN/FAIL (run_gate)",
          responses={200: {"content": {"application/json": {"example": _EVALUATE_EXAMPLE}}},
                     404: {"description": "run directory or attribution.jsonl not found"},
                     422: {"description": "missing/mistyped field or malformed artifact"}})
def evaluate_endpoint(req: EvaluateRequest) -> GateResult:
    """Compare a **candidate** RAG run against a **baseline** and detect quality
    regressions by failure mode, with bootstrap confidence intervals.

    Each input is a directory holding `attribution.jsonl` (the baseline also a
    `noise_band.json`) — the SAME contract as the `run_gate` CLI/MCP, so the same
    input gives the same verdict across all three interfaces.

    Returns `verdict` (PASS/WARN/FAIL), `exit_code` (FAIL=1), significant
    `regressions`, borderline `warnings`, a failure-mode `diagnosis`, and rule-based
    `suggestions` (suggestion-only — nothing is applied).

    Errors: **422** missing/mistyped field or malformed artifact · **404** dir/file absent."""
    _require_artifacts(req.baseline_dir, ("attribution.jsonl",))
    _require_artifacts(req.candidate_dir, ("attribution.jsonl",))
    return build_gate_result(req.baseline_dir, req.candidate_dir)


@app.post("/analyze", response_model=FailureAnalysis, tags=["analyze"],
          summary="Diagnose one run → bottleneck & priorities (analyze_failures)",
          responses={200: {"content": {"application/json": {"example": _ANALYZE_EXAMPLE}}},
                     404: {"description": "run directory or attribution.jsonl not found"},
                     422: {"description": "missing field or malformed artifact"}})
def analyze_endpoint(req: AnalyzeRequest) -> FailureAnalysis:
    """Diagnose a **single** run: where is it weak and what to fix first?

    Takes one directory holding `attribution.jsonl` (same contract as the
    `analyze_failures` MCP tool). Returns the failure distribution, the bottleneck
    pipeline stage (retrieval / grounding / …), per-slice failure rates, a
    groundedness breakdown, deterministic RAGAS-equivalent metrics (no LLM judge),
    and improvement priorities (suggestion-only — re-verify with /evaluate).

    Errors: **422** missing field or malformed artifact · **404** dir/file absent."""
    _require_artifacts(req.run_dir, ("attribution.jsonl",))
    return build_analysis(req.run_dir)
