"""API-level tests for the REST interface (phase A2): happy path + error handling.

The API is a thin adapter over the shared core (app/core), so these tests focus on
the HTTP contract — status codes, error bodies, and that the happy path returns the
same verdict the core does — not on re-testing the statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.service import build_gate_result

client = TestClient(app)

BASE = "examples/allganize_baseline"
CAND = "examples/allganize_candidate"


# --- happy path ------------------------------------------------------------- #

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_redirects_to_docs():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/docs"


def test_evaluate_matches_core():
    r = client.post("/evaluate", json={"baseline_dir": BASE, "candidate_dir": CAND})
    assert r.status_code == 200
    assert r.json() == build_gate_result(BASE, CAND).model_dump()  # API == core


def test_analyze_ok():
    r = client.post("/analyze", json={"run_dir": BASE})
    assert r.status_code == 200
    assert r.json()["bottleneck_stage"] == "grounding"


# --- error handling --------------------------------------------------------- #

def test_evaluate_missing_field_422():
    r = client.post("/evaluate", json={"baseline_dir": BASE})  # candidate_dir missing
    assert r.status_code == 422


def test_evaluate_missing_dir_404():
    r = client.post("/evaluate", json={"baseline_dir": BASE, "candidate_dir": "examples/nope"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_analyze_missing_artifact_404(tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()  # dir exists but has no attribution.jsonl
    r = client.post("/analyze", json={"run_dir": str(empty)})
    assert r.status_code == 404
    assert "missing artifact" in r.json()["detail"]


def test_analyze_malformed_artifact_422(tmp_path):
    bad = tmp_path / "bad_run"
    bad.mkdir()
    (bad / "attribution.jsonl").write_text("this is not json\n", encoding="utf-8")
    r = client.post("/analyze", json={"run_dir": str(bad)})
    assert r.status_code == 422
    assert "malformed" in r.json()["detail"]


def test_analyze_missing_field_422():
    r = client.post("/analyze", json={})  # run_dir missing
    assert r.status_code == 422
