"""Per-case gate fields — precomputed upstream so the CI gate needs no gold.

The gate only needs booleans per case (answerable / correct / value_present /
retrieval_strict_ok / mode). These are DETERMINISTIC functions of gold + run-log,
so we compute them once upstream (where gold is available) and store them in the
attribution artifact. The gate then reads them directly — no eval_cases, no LLM,
no embedding.
"""

from __future__ import annotations

from app.evaluator.attribution import is_retrieval_miss
from app.evaluator.metrics import value_present as _dart_value_present


def gate_fields(case: dict, rec: dict, primary_failure: str,
                matcher=None, value_present=None) -> dict:
    """Self-contained per-case gate input (folded into the attribution record).

    matcher / value_present default to the DART implementations (so existing DART
    callers stay byte-identical). A domain (e.g. Wiki) injects its GoldMatcher and
    value-presence hook here; the downstream engine (detect/gate/metrics) is unchanged.
    """
    vp = value_present if value_present is not None else _dart_value_present
    answerable = case["answer_type"] == "answerable"
    correct = primary_failure == "correct"
    return {
        "id": case["id"],
        "slice": case["slice"],
        "answerable": answerable,
        "correct": correct,
        "value_present": vp(case, rec),
        "retrieval_strict_ok": answerable and not is_retrieval_miss(case, rec, matcher),
        "over_answer": (not answerable) and (not correct),
        "primary_failure": primary_failure,
    }
