"""Single source of truth for shared data schemas.

Per CLAUDE.md, the failure taxonomy and the EvalCase model are defined here once
and imported everywhere else. Do not redefine these elsewhere.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FailureType(StrEnum):
    """Closed failure taxonomy (CLAUDE.md §3). New modes require human approval."""

    CORRECT = "correct"
    RETRIEVAL_MISS = "retrieval_miss"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    TABLE_VALUE_ERROR = "table_value_error"
    WRONG_PERIOD = "wrong_period"
    UNIT_ERROR = "unit_error"
    HALLUCINATION = "hallucination"
    REASONING_ERROR = "reasoning_error"
    FORMAT_VIOLATION = "format_violation"


AnswerType = Literal["answerable", "unanswerable"]   # binary, domain-agnostic
# Relaxed to str for multi-domain (interfaces §0). Type safety is enforced at
# RUNTIME by each EvalSetProvider's whitelist (allowed_slices / allowed_schemas) —
# so a typo like "texr" is caught there, not silently accepted.
Slice = str
AnswerSchema = str


# --------------------------------------------------------------------------- #
# expected_answer shapes (one per answer_schema). extra="forbid" keeps the
# Union unambiguous: a malformed payload cannot silently fall through to NoAnswer.
# --------------------------------------------------------------------------- #

class NumericAnswer(BaseModel):
    """table_value / numeric_reasoning (single company)."""

    model_config = ConfigDict(extra="forbid")
    value: int | float
    unit: str
    basis: str  # e.g. "연결" / "별도"
    period: str  # e.g. "2023(제55기)"


class CompanyValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int | float
    unit: str


class ComparisonAnswer(BaseModel):
    """numeric_reasoning (cross-company comparison).

    NOTE: companies are nested under `companies` (not flat top-level keys) so the
    payload validates cleanly and `comparison` cannot collide with a company name.
    """

    model_config = ConfigDict(extra="forbid")
    companies: dict[str, CompanyValue]
    comparison: str  # e.g. "삼성전자 > 하이닉스"


class TextAnswer(BaseModel):
    """body_text — reference key points for judge-based semantic scoring."""

    model_config = ConfigDict(extra="forbid")
    key_points: list[str]


class NoAnswer(BaseModel):
    """no_answer — sentinel; model should refuse rather than fabricate."""

    model_config = ConfigDict(extra="forbid")
    sentinel: Literal["정보 없음"] = "정보 없음"


ExpectedAnswer = NumericAnswer | ComparisonAnswer | TextAnswer | NoAnswer

_SCHEMA_TO_ANSWER: dict[str, type[BaseModel]] = {
    "numeric": NumericAnswer,
    "comparison": ComparisonAnswer,
    "text": TextAnswer,
    "no_answer": NoAnswer,
}


class EvalCase(BaseModel):
    """One fixed evaluation case (plan-v3 §5.3, Phase 1.2).

    All Phase 1.2 cases are should-pass, so gold_failure_type is always "correct".
    """

    id: str
    company: str
    source_doc: str
    fiscal_year: int
    question: str
    contexts: list[str] = Field(default_factory=list)  # gold evidence, NOT model input
    answer_schema: AnswerSchema
    expected_answer: ExpectedAnswer
    answer_type: AnswerType
    slice: Slice
    gold_failure_type: str  # taxonomy value; FailureType enum lists DART/engine modes
    source_ref: str  # file / table-id / cell, or paragraph the answer came from
    needs_review: bool = False

    @model_validator(mode="after")
    def _expected_answer_matches_schema(self) -> "EvalCase":
        expected_type = _SCHEMA_TO_ANSWER.get(self.answer_schema)
        if expected_type and not isinstance(self.expected_answer, expected_type):
            raise ValueError(
                f"answer_schema={self.answer_schema!r} requires "
                f"{expected_type.__name__}, got {type(self.expected_answer).__name__}"
            )
        # Unknown (domain) schema: expected_answer is still constrained by the
        # ExpectedAnswer Union; slice/answer_schema validity is the provider whitelist's job.
        return self
