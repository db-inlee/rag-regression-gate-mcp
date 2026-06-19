"""Plugin interfaces for domain portability (design = docs/interfaces.md).

Engine code depends on these Protocols, not on DART specifics. The DART reference
implementations (app/adapters/dart.py) are PURE DELEGATION to existing functions —
no behavior change (verified by the gate: same config → diff 0).

A second domain (e.g. English Wiki QA) implements these four Protocols (+ the two
secondary hooks: Value-Presence, Failure Taxonomy) and the engine runs unchanged.
"""

from __future__ import annotations

from typing import Hashable, Protocol, TypedDict, runtime_checkable

from app.schemas import EvalCase, ExpectedAnswer


# --- shared types (typed names for the dict shapes already in use) --------- #

class ScoreResult(TypedDict):
    correct: bool | None        # None = deferred to judge (narrative answers)
    score_detail: dict


class RetrievedChunk(TypedDict):
    text: str
    metadata: dict              # domain-free: {source_file, table_id, page, ...} | {doc_id, section, ...}


class RunLogEntry(TypedDict):
    id: str
    slice: str
    question: str
    retrieved_chunks: list      # list[RetrievedChunk]
    answer: str | None
    latency_ms: float
    llm_calls: int
    token_usage: dict


GoldRef = Hashable              # domain-defined evidence key (DART: a tuple)


# --- the four domain plugins ----------------------------------------------- #

@runtime_checkable
class ScoringPlugin(Protocol):
    def score(self, answer: str | None, gold: ExpectedAnswer, case: EvalCase) -> ScoreResult: ...
    def is_refusal(self, answer: str) -> bool: ...


@runtime_checkable
class GoldMatcher(Protocol):
    def gold_refs(self, case: EvalCase) -> set[GoldRef]: ...
    def retrieved_refs(self, entry: RunLogEntry) -> set[GoldRef]: ...
    # engine composes: is_retrieved = gold_refs(case) <= retrieved_refs(entry)


@runtime_checkable
class EvalSetProvider(Protocol):
    def load(self) -> list[EvalCase]: ...


@runtime_checkable
class RAGAdapter(Protocol):
    def run(self, question: str) -> RunLogEntry: ...
