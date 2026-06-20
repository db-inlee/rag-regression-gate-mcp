"""Allganize law/public — the THIRD implementation of app.interfaces (external gold).

Unlike DART (we built the eval set) and Wiki (we excerpted SQuAD), this domain runs
the engine on an EXTERNAL Korean gold dataset we did NOT author (datalama/
RAG-Evaluation-Dataset-KO). Engine (detect/gate/metrics/noise_band) is untouched;
only the four plugins + two hooks are added here.

Key portability evidence — the gold-evidence UNIT differs per domain:
  DART      : page / table-cell  (fine-grained)
  Wiki      : paragraph (doc_id)
  Allganize : DOCUMENT (pid)     ← coarser

So only the GoldMatcher's KEY changes (page -> document); the ⊆ logic
(gold_retrieved) is the SAME shared engine code. Honest limitation: document-level
matching is COARSER than DART — retrieving a *different page of the same document*
still counts as retrieval success. groundedness here is likewise document-level
(answer is "grounded" iff the gold document was retrieved), for the same reason.

ScoringPlugin is REUSED from DART (same Korean: unit normalization + refusal
phrases; text schema defers to the shared judge). EvalSetProvider + RAGAdapter +
GoldMatcher are new.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from app.adapters.dart import DartScoringPlugin
from app.config import RagConfig
from app.interfaces import RunLogEntry, check_whitelist
from app.schemas import EvalCase

EXCERPT_PATH = Path("data/allganize_eval/allganize_excerpt.jsonl")
CORPUS_PATH = Path("data/allganize_eval/allganize_corpus.jsonl")  # all law+public pages (distractors)

# Same engine/config model as DART; bge-m3 (Korean) like DART, but an isolated
# collection_base/persist_dir so the index_signature keeps it separate from DART.
ALLGANIZE_CONFIG = RagConfig(
    embedding_model="BAAI/bge-m3",  # Korean, same as DART
    chunk_size=1000,
    chunk_overlap=150,
    top_k=5,
    collection_base="allganize",
    persist_dir="data/allganize_index",
)


# --------------------------------------------------------------------------- #
# D2 — EvalSetProvider #3 (external gold: test_cases -> EvalCase)
# --------------------------------------------------------------------------- #

class AllganizeEvalSetProvider:
    """→ data/allganize_eval/allganize_excerpt.jsonl (40 cases) → EvalCase.

    Mapping (EvalCase schema unchanged — fields only):
      question                         → question
      target_answer                    → expected_answer=TextAnswer(key_points=[...])
      context_type ("paragraph"/"table") → slice
      pid (gold DOCUMENT)              → source_ref   (gold evidence key, doc-level)
      file_name                        → source_doc
      answer_schema="text", answer_type="answerable", gold_failure_type="correct".
    """

    ALLOWED_SLICES = {"paragraph", "table"}
    ALLOWED_SCHEMAS = {"text"}

    def __init__(self, path: str | Path = EXCERPT_PATH):
        self.path = Path(path)

    def _to_case(self, i: int, row: dict) -> EvalCase:
        return EvalCase.model_validate({
            "id": f"allganize_{i:03d}",
            "company": row["domain"],          # free field (law/public)
            "source_doc": row["file_name"],
            "fiscal_year": 0,                   # N/A
            "question": row["question"],
            "contexts": [],                    # gold text not bundled; matching is doc-level (pid)
            "answer_schema": "text",
            "expected_answer": {"key_points": [row["target_answer"]]},
            "answer_type": "answerable",
            "slice": row["context_type"],
            "gold_failure_type": "correct",
            "source_ref": row["pid"],          # gold DOCUMENT id (coarser than DART)
            "needs_review": False,
        })

    def load(self) -> list[EvalCase]:
        rows = [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cases = [self._to_case(i + 1, r) for i, r in enumerate(rows)]
        check_whitelist(cases, self.ALLOWED_SLICES, self.ALLOWED_SCHEMAS)
        return cases


# --------------------------------------------------------------------------- #
# D2 — GoldMatcher #3 : DOCUMENT-level (pid). ⊆ logic reused from the engine.
# --------------------------------------------------------------------------- #

class AllganizeGoldMatcher:
    """gold_refs={pid}, retrieved_refs={chunk.pid}. Coarser than DART (document, not
    page): a different page of the gold document still counts as retrieved. The
    engine's gold_refs ⊆ retrieved_refs judgement is the SAME shared code."""

    def gold_refs(self, case) -> set:
        c = case if isinstance(case, dict) else case.model_dump()
        ref = c.get("source_ref")
        return {ref} if ref else set()

    def retrieved_refs(self, entry: RunLogEntry) -> set:
        return {ch["metadata"].get("pid") for ch in entry.get("retrieved_chunks", [])
                if ch.get("metadata", {}).get("pid")}


def allganize_value_present(case, rec) -> bool:
    """Anti-illusion hook (document-level, coarser than DART's value-in-context).

    A narrative Korean answer has no single 'value' to substring-match, and our
    gold unit is the document — so groundedness here = 'was the gold document
    actually retrieved?'. correct & value_present → grounded; correct & not →
    unsupported (right answer without retrieving the gold doc = memorization/luck)."""
    c = case if isinstance(case, dict) else case.model_dump()
    ref = c.get("source_ref")
    if not ref:
        return False
    retrieved = {ch["metadata"].get("pid") for ch in rec.get("retrieved_chunks", [])
                 if ch.get("metadata", {}).get("pid")}
    return ref in retrieved


# --------------------------------------------------------------------------- #
# D2 — RAGAdapter #3 : bge-m3 (Korean) + gpt-4o-mini, document-level pid metadata
# --------------------------------------------------------------------------- #

# Reuse DART's Korean system prompt (same generation contract).
from app.rag.pipeline import SYSTEM_PROMPT as _SYSTEM_KO


def _corpus_pages() -> list[dict]:
    """Non-empty law+public pages to index (38 image-only pages are skipped)."""
    path = CORPUS_PATH if CORPUS_PATH.exists() else EXCERPT_PATH
    pages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("text", "").strip():
            pages.append(r)
    return pages


def build_allganize_index(config: RagConfig = ALLGANIZE_CONFIG, rebuild: bool = False) -> int:
    """Chunk law+public pages and index them in an isolated Chroma collection
    (bge-m3), keyed by the config signature. Chunk metadata carries (pid, file_name,
    page) so AllganizeGoldMatcher can compare gold/retrieved at document level."""
    import chromadb
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.rag.index import _collection_exists, _embeddings

    name = config.collection_name()
    existing = _collection_exists(config)
    if existing > 0 and not rebuild:
        return existing

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    docs = []
    for p in _corpus_pages():
        for piece in splitter.split_text(p["text"]):
            docs.append(Document(page_content=piece, metadata={
                "pid": p["pid"], "doc_id": p["pid"], "file_name": p["file_name"],
                "page": p["page_number"], "domain": p["domain"],
                "source_file": p["file_name"]}))

    persist = Path(config.persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    if rebuild:
        client = chromadb.PersistentClient(path=str(persist))
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)
    Chroma.from_documents(
        documents=docs, embedding=_embeddings(config),
        collection_name=name, persist_directory=str(persist),
        collection_metadata={"index_signature": config.index_signature(),
                             "embedding_model": config.embedding_model, "n_chunks": len(docs)})
    return len(docs)


class AllganizeRAGAdapter:
    """→ allganize Chroma index (bge-m3) + gpt-4o-mini generation. Lazy index load."""

    def __init__(self, config: RagConfig = ALLGANIZE_CONFIG):
        self.config = config
        self._store = None
        self._client = None

    def _ensure(self):
        if self._store is None:
            from langchain_chroma import Chroma
            from openai import OpenAI

            from app.env import require_openai_key
            from app.rag.index import _collection_exists, _embeddings

            if _collection_exists(self.config) <= 0:
                build_allganize_index(self.config)
            self._store = Chroma(collection_name=self.config.collection_name(),
                                 persist_directory=self.config.persist_dir,
                                 embedding_function=_embeddings(self.config))
            require_openai_key()
            self._client = OpenAI()
        return self._store

    def run(self, question: str) -> RunLogEntry:
        store = self._ensure()
        t0 = time.perf_counter()
        hits = store.similarity_search_with_score(question, k=self.config.top_k)
        chunks = [{"text": doc.page_content, "metadata": dict(doc.metadata),
                   "distance": float(score)} for doc, score in hits]

        context = "\n\n".join(
            f"[{i}] ({c['metadata'].get('file_name')} p{c['metadata'].get('page')})\n{c['text']}"
            for i, c in enumerate(chunks, 1)) or "(컨텍스트 없음)"
        resp = self._client.chat.completions.create(
            model=self.config.generation_model, temperature=0, seed=self.config.seed,
            messages=[{"role": "system", "content": _SYSTEM_KO},
                      {"role": "user", "content": f"컨텍스트:\n{context}\n\n질문: {question}\n답변:"}])
        u = resp.usage
        return {
            "question": question,
            "retrieved_chunks": chunks,
            "answer": (resp.choices[0].message.content or "").strip(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "llm_calls": 1,
            "token_usage": {"prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens},
        }


# --------------------------------------------------------------------------- #
# Orchestration glue (reuses the SHARED attribution/gate/judge engine)
# --------------------------------------------------------------------------- #

def evaluate_allganize(config: RagConfig = ALLGANIZE_CONFIG, client=None, cases: dict | None = None):
    """One full allganize eval pass: RAG run → score (DART scorer; text→judge) →
    attribute → gate_fields. Returns (cases, records, attrs).

    The attribution/gate_fields/judge calls are the SAME engine functions DART uses —
    only AllganizeGoldMatcher + allganize_value_present are injected."""
    from openai import OpenAI

    from app.env import require_openai_key
    from app.evaluator.attribution import attribute
    from app.evaluator.case_eval import gate_fields
    from app.evaluator.judge import judge_body_text

    if cases is None:
        cases = {c.model_dump()["id"]: c.model_dump() for c in AllganizeEvalSetProvider().load()}
    if client is None:
        require_openai_key()
        client = OpenAI()

    adapter = AllganizeRAGAdapter(config)
    scorer, matcher = DartScoringPlugin(), AllganizeGoldMatcher()
    records, attrs = [], []
    for cid, case in cases.items():
        entry = adapter.run(case["question"])
        records.append({"type": "case", "id": cid, "slice": case["slice"], **entry})

        scored = scorer.score(entry["answer"], case["expected_answer"], case)
        correct = scored["correct"]
        if correct is None:  # text → judge (reuse DART judge on key_points)
            v = judge_body_text(case["question"], case["expected_answer"]["key_points"],
                                entry["answer"], config, client)
            correct = v["correct"]
            scored["score_detail"]["judge_reason"] = v["reason"]

        a = attribute(case, correct, scored, entry, config, client=client, matcher=matcher)
        a.update(gate_fields(case, entry, a["primary_failure"],
                             matcher=matcher, value_present=allganize_value_present))
        attrs.append(a)
    return cases, records, attrs


def allganize_metrics(attrs: list[dict]) -> dict:
    """Headline metrics from the portable gate_fields (same DEFINITIONS as DART)."""
    ans = [g for g in attrs if g["answerable"]]
    na = [g for g in attrs if not g["answerable"]]
    grounded = [g for g in ans if g["correct"] and g["value_present"]]
    return {
        "answerable_accuracy": round(len(grounded) / len(ans), 4) if ans else None,
        "answerable_total": len(ans),
        "grounded_correct": len(grounded),
        "unsupported_correct": sum(1 for g in ans if g["correct"] and not g["value_present"]),
        "no_answer_accuracy": round(sum(g["correct"] for g in na) / len(na), 4) if na else None,
        "over_answer_rate": round(sum(g["over_answer"] for g in na) / len(na), 4) if na else None,
        "retrieval_success_strict": round(sum(g["retrieval_strict_ok"] for g in ans) / len(ans), 4) if ans else None,
        "retrieval_miss_count": sum(1 for g in attrs if g["primary_failure"] == "retrieval_miss"),
        "failure_distribution": dict(Counter(g["primary_failure"] for g in attrs)),
    }


def allganize_band_vector(attrs: list[dict]) -> dict:
    """Per-run metric vector for noise-band measurement (keys = detect noise_keys)."""
    ans = [g for g in attrs if g["answerable"]]
    na = [g for g in attrs if not g["answerable"]]
    rate = lambda sub, f: (sum(f(g) for g in sub) / len(sub)) if sub else 0.0
    return {
        "answerable_accuracy": rate(ans, lambda g: g["correct"] and g["value_present"]),
        "retrieval_success_strict": rate(ans, lambda g: g["retrieval_strict_ok"]),
        "retrieval_success_value_present": rate(ans, lambda g: g["value_present"]),
        "no_answer_accuracy": rate(na, lambda g: g["correct"]),
        "over_answer_rate": rate(na, lambda g: g["over_answer"]),
        "mode:retrieval_miss": sum(g["primary_failure"] == "retrieval_miss" for g in attrs),
        "mode:hallucination": sum(g["primary_failure"] == "hallucination" for g in attrs),
    }
