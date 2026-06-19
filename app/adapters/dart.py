"""DART reference implementations of app.interfaces — PURE DELEGATION.

Each method only calls an existing function (scorer.score_case, attribution.gold_keys/
retrieved_keys, eval_cases.jsonl load, pipeline.Pipeline.run). No new logic, so the
DART pipeline's behavior is unchanged — verified by the gate (same config → diff 0).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import DEFAULT_CONFIG, RagConfig
from app.evaluator import attribution, scorer
from app.interfaces import RunLogEntry, ScoreResult
from app.schemas import EvalCase


def _as_dict(case) -> dict:
    return case if isinstance(case, dict) else case.model_dump()


class DartScoringPlugin:
    """→ app/evaluator/scorer.py (한국어 단위 정규화·±0.1%·comparison·거부문구)."""

    def score(self, answer: str | None, gold, case) -> ScoreResult:
        return scorer.score_case(_as_dict(case), answer)

    def is_refusal(self, answer: str) -> bool:
        return scorer.is_refusal(answer or "")


class DartGoldMatcher:
    """→ app/evaluator/attribution.py (source_ref 파싱 + 청크 메타 매칭)."""

    def gold_refs(self, case) -> set:
        return set(attribution.gold_keys(_as_dict(case)))

    def retrieved_refs(self, entry: RunLogEntry) -> set:
        return attribution.retrieved_keys(entry)


class DartEvalSetProvider:
    """→ data/eval_cases.jsonl (고정 평가셋 로딩)."""

    def __init__(self, path: str | Path = "data/eval_cases.jsonl"):
        self.path = Path(path)

    def load(self) -> list[EvalCase]:
        return [EvalCase.model_validate(json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DartRAGAdapter:
    """→ app/rag/pipeline.py Pipeline (Chroma + bge-m3 + gpt-4o-mini). 인덱스는 lazy 로드."""

    def __init__(self, config: RagConfig = DEFAULT_CONFIG):
        self.config = config
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            from app.rag.pipeline import Pipeline
            self._pipe = Pipeline(self.config)
        return self._pipe

    def run(self, question: str) -> RunLogEntry:
        r = self._pipeline().run(question)
        return {
            "question": question,
            "retrieved_chunks": r["retrieved_chunks"],
            "answer": r["answer"],
            "latency_ms": r["latency_ms"],
            "llm_calls": r["llm_calls"],
            "token_usage": r["token_usage"],
        }
