"""Tests for the judge's JSON parsing + graceful fallback (no API needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluator.judge import _parse_verdict


def test_parse_valid_verdict():
    v, ok = _parse_verdict('{"correct": true, "reason": "핵심 포인트 충족"}')
    assert ok is True and v["correct"] is True and "핵심" in v["reason"]


def test_parse_valid_incorrect():
    v, ok = _parse_verdict('{"correct": false, "reason": "숫자 불일치"}')
    assert ok is True and v["correct"] is False


def test_parse_malformed_falls_back_without_crashing():
    for bad in ["not json at all", "", "{correct: true}", '{"reason": "no correct field"}']:
        v, ok = _parse_verdict(bad)
        assert ok is False
        assert v["correct"] is False  # fallback is conservative (incorrect)
        assert "parse_failed" in v["reason"]
