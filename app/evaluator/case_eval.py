"""Per-case gate fields — precomputed upstream so the CI gate needs no gold.

The gate only needs booleans per case (answerable / correct / value_present /
retrieval_strict_ok / mode). These are DETERMINISTIC functions of gold + run-log,
so we compute them once upstream (where gold is available) and store them in the
attribution artifact. The gate then reads them directly — no eval_cases, no LLM,
no embedding.
"""

from __future__ import annotations

from app.evaluator.attribution import is_retrieval_miss
from app.evaluator.metrics import value_present


def gate_fields(case: dict, rec: dict, primary_failure: str) -> dict:
    """Self-contained per-case gate input (folded into the attribution record)."""
    answerable = case["answer_type"] == "answerable"
    correct = primary_failure == "correct"
    return {
        "id": case["id"],
        "slice": case["slice"],
        "answerable": answerable,
        "correct": correct,
        "value_present": value_present(case, rec),
        "retrieval_strict_ok": answerable and not is_retrieval_miss(case, rec),
        "over_answer": (not answerable) and (not correct),
        "primary_failure": primary_failure,
    }
