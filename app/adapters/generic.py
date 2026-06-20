"""Config-driven domain adapters (phase B) — generalize the per-domain plugin
classes into INJECTED key-extraction functions / column maps.

Pure addition: the engine (detect/gate/metrics/attribution) and the existing
dart/wiki/allganize adapters are untouched. Generic reproduces them via config and
proves diff-0 against the dedicated classes (same verification style as before).

B1 (this step): GenericGoldMatcher — what actually differs per domain is only
"where the gold key lives / where the retrieved key lives"; the ⊆ comparison
(gold_refs ⊆ retrieved_refs) stays in the shared engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.interfaces import GoldRef, RunLogEntry, check_whitelist
from app.schemas import EvalCase


class GenericGoldMatcher:
    """Generalize a domain GoldMatcher to TWO key-extraction functions.

    gold_key:      case-dict   -> set[GoldRef]
    retrieved_key: RunLogEntry -> set[GoldRef]

    The engine's ⊆ judgement (gold_refs ⊆ retrieved_refs via gold_retrieved) is
    unchanged — this class only defines HOW the keys are pulled. Satisfies the
    GoldMatcher Protocol (gold_refs / retrieved_refs)."""

    def __init__(self,
                 gold_key: Callable[[dict], set[GoldRef]],
                 retrieved_key: Callable[[RunLogEntry], set[GoldRef]]):
        self._gold = gold_key
        self._retrieved = retrieved_key

    def gold_refs(self, case) -> set:
        c = case if isinstance(case, dict) else case.model_dump()
        return self._gold(c)

    def retrieved_refs(self, entry: RunLogEntry) -> set:
        return self._retrieved(entry)


# --------------------------------------------------------------------------- #
# Reproduction lambdas — existing domains expressed as GenericGoldMatcher config.
# These return matchers IDENTICAL in behavior to the dedicated classes.
# --------------------------------------------------------------------------- #

def allganize_gold_matcher() -> GenericGoldMatcher:
    """Reproduce AllganizeGoldMatcher (document-level: pid)."""
    return GenericGoldMatcher(
        gold_key=lambda c: {c["source_ref"]} if c.get("source_ref") else set(),
        retrieved_key=lambda e: {ch["metadata"].get("pid")
                                 for ch in e.get("retrieved_chunks", [])
                                 if ch.get("metadata", {}).get("pid")},
    )


def wiki_gold_matcher() -> GenericGoldMatcher:
    """Reproduce WikiGoldMatcher (paragraph-level: doc_id)."""
    return GenericGoldMatcher(
        gold_key=lambda c: {c["source_ref"]} if c.get("source_ref") else set(),
        retrieved_key=lambda e: {ch["metadata"].get("doc_id")
                                 for ch in e.get("retrieved_chunks", [])
                                 if ch.get("metadata", {}).get("doc_id")},
    )


# --------------------------------------------------------------------------- #
# B2 — GenericValuePresence: the common "gold ref ∈ retrieved" groundedness hook.
# allganize_value_present / wiki value_present are both this pattern; reusing the
# SAME key extraction as the GoldMatcher collapses them to one factory.
# (DART's number-in-context precision matching stays domain-specific — by design.)
# --------------------------------------------------------------------------- #

def make_value_present(matcher: GenericGoldMatcher):
    """Document/structure-level groundedness = did the gold ref actually get retrieved?

    Returns a value_present(case, rec) -> bool hook, injectable into gate_fields
    exactly like the dedicated allganize_value_present / wiki_value_present."""
    def value_present(case, rec) -> bool:
        gold = matcher.gold_refs(case)
        retrieved = matcher.retrieved_refs(rec)
        return bool(gold) and gold <= retrieved
    return value_present


# --------------------------------------------------------------------------- #
# B3 — GenericEvalSetProvider + ColumnMap: "jsonl field -> EvalCase field" as
# config instead of a hand-written _to_case per domain. Adds a new domain with
# NO new class — just a column map.
# --------------------------------------------------------------------------- #

@dataclass
class ColumnMap:
    """Which jsonl COLUMN feeds each EvalCase field (per-row), plus fixed constants.

    question/answer/slice/gold_ref are per-row column names. source_doc/company are
    optional per-row column names (None -> "" / default). `extra` holds truly FIXED
    constants (e.g. fiscal_year=0) merged verbatim into every case."""

    question: str
    answer: str                       # → expected_answer.key_points = [row[answer]]
    slice: str
    gold_ref: str                     # → source_ref
    source_doc: str | None = None     # column name (per-row), else ""
    company: str | None = None        # column name (per-row), e.g. Allganize "domain"
    answer_schema: str = "text"
    answer_type: str = "answerable"
    extra: dict = field(default_factory=dict)  # fixed constants (fiscal_year=0, …)


class GenericEvalSetProvider:
    """jsonl + ColumnMap → list[EvalCase]. Add a new domain with config, no class.

    Satisfies the EvalSetProvider Protocol (load). Whitelist check is identical to
    the dedicated providers (type-safety net for the relaxed slice/answer_schema)."""

    def __init__(self, path, colmap: ColumnMap,
                 allowed_slices: set[str], allowed_schemas: set[str],
                 id_prefix: str = "case"):
        self.path = Path(path)
        self.cmap = colmap
        self.allowed_slices = allowed_slices
        self.allowed_schemas = allowed_schemas
        self.id_prefix = id_prefix

    def _to_case(self, i: int, row: dict) -> EvalCase:
        m = self.cmap
        return EvalCase.model_validate({
            "id": f"{self.id_prefix}_{i:03d}",
            "question": row[m.question],
            "expected_answer": {"key_points": [row[m.answer]]},
            "answer_schema": m.answer_schema,
            "answer_type": m.answer_type,
            "slice": row[m.slice],
            "source_ref": row[m.gold_ref],
            "source_doc": row[m.source_doc] if m.source_doc else "",
            "company": row[m.company] if m.company else "",
            "gold_failure_type": "correct",
            "contexts": [],
            "needs_review": False,
            **m.extra,          # fixed constants (e.g. fiscal_year=0)
        })

    def load(self) -> list[EvalCase]:
        rows = [json.loads(l) for l in
                self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cases = [self._to_case(i + 1, r) for i, r in enumerate(rows)]
        check_whitelist(cases, self.allowed_slices, self.allowed_schemas)
        return cases


def allganize_eval_provider(path="data/allganize_eval/allganize_excerpt.jsonl") -> GenericEvalSetProvider:
    """Reproduce AllganizeEvalSetProvider purely as config (company ← row["domain"],
    fiscal_year fixed 0). Same EvalCase list as the dedicated class (diff 0)."""
    return GenericEvalSetProvider(
        path=path,
        colmap=ColumnMap(question="question", answer="target_answer",
                         slice="context_type", gold_ref="pid",
                         source_doc="file_name", company="domain",
                         extra={"fiscal_year": 0}),
        allowed_slices={"paragraph", "table"}, allowed_schemas={"text"},
        id_prefix="allganize",
    )
