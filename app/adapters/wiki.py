"""Wiki (English SQuAD 2.0) mini-domain — the SECOND implementation of app.interfaces.

Proves the four plugin interfaces are domain-agnostic: this file adds an English
Wikipedia-QA domain WITHOUT touching DART code or the engine (detect/gate/metrics/
noise_band). DART stays the main reference; this is a small honest instance (20 cases).

G2.1 implements EvalSetProvider here. G2.2-G2.4 (RAGAdapter / ScoringPlugin /
GoldMatcher) are added later in this same file.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.interfaces import check_whitelist
from app.schemas import EvalCase

EXCERPT_PATH = Path("data/wiki_eval/squad2_excerpt.jsonl")


class WikiEvalSetProvider:
    """→ data/wiki_eval/squad2_excerpt.jsonl (SQuAD 2.0 dev 발췌 → EvalCase).

    Mapping (EvalCase schema unchanged — fields only):
      factoid (is_impossible=false) → answer_schema="text",  answer_type="answerable",
          slice="factoid",   expected_answer=TextAnswer(key_points=[gold answers])
      no_answer (is_impossible=true) → answer_schema="no_answer", answer_type="unanswerable",
          slice="no_answer", expected_answer=NoAnswer()
      context paragraph → contexts (gold evidence) ; doc_id ("Title#pN") → source_ref
      gold_failure_type = "correct" (all should-pass, same convention as DART).
    """

    ALLOWED_SLICES = {"factoid", "no_answer"}
    ALLOWED_SCHEMAS = {"text", "no_answer"}

    def __init__(self, path: str | Path = EXCERPT_PATH):
        self.path = Path(path)

    def _to_case(self, i: int, row: dict) -> EvalCase:
        impossible = row["is_impossible"]
        doc_id = row["doc_id"]
        common = {
            "id": f"wiki_{i:03d}",
            "company": row["title"],              # domain-agnostic free field (article subject)
            "source_doc": f"{row['title']}.txt",
            "fiscal_year": 0,                     # N/A for wiki
            "question": row["question"],
            "contexts": [row["context"]],         # gold evidence paragraph
            "gold_failure_type": "correct",
            "source_ref": doc_id,                 # wiki evidence key (NOT DART table-ref format)
            "needs_review": False,
        }
        if impossible:
            return EvalCase.model_validate({
                **common,
                "answer_schema": "no_answer",
                "expected_answer": {"sentinel": "정보 없음"},
                "answer_type": "unanswerable",
                "slice": "no_answer",
            })
        return EvalCase.model_validate({
            **common,
            "answer_schema": "text",
            "expected_answer": {"key_points": row["answers"]},
            "answer_type": "answerable",
            "slice": "factoid",
        })

    def load(self) -> list[EvalCase]:
        rows = [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cases = [self._to_case(i + 1, r) for i, r in enumerate(rows)]
        check_whitelist(cases, self.ALLOWED_SLICES, self.ALLOWED_SCHEMAS)  # type-safety net
        return cases
