"""Wiki (English SQuAD 2.0) mini-domain — the SECOND implementation of app.interfaces.

Proves the four plugin interfaces are domain-agnostic: this file adds an English
Wikipedia-QA domain WITHOUT touching DART code or the engine (detect/gate/metrics/
noise_band). DART stays the main reference; this is a small honest instance (20 cases).

G2.1 implements EvalSetProvider here. G2.2-G2.4 (RAGAdapter / ScoringPlugin /
GoldMatcher) are added later in this same file.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.config import RagConfig
from app.interfaces import RunLogEntry, ScoreResult, check_whitelist
from app.schemas import EvalCase

EXCERPT_PATH = Path("data/wiki_eval/squad2_excerpt.jsonl")

# Wiki config: SAME engine/config model as DART, only the swap-fields differ.
# A different embedding (MiniLM, English) + collection_base/persist_dir isolate the
# index from DART via index_signature — proving "RAG impl differs, engine identical".
WIKI_CONFIG = RagConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # ≠ DART bge-m3
    chunk_size=400,
    chunk_overlap=50,
    top_k=3,
    collection_base="wiki",
    persist_dir="data/wiki_index",
)


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


# --------------------------------------------------------------------------- #
# G2.3 — English scoring + gold matching (ScoringPlugin / GoldMatcher) #2
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
# English refusal phrases (the wiki counterpart of scorer._REFUSAL).
_REFUSAL_EN = ("not stated", "no answer", "cannot be answered", "can't be answered",
               "not mentioned", "not provided", "does not contain", "doesn't contain",
               "no information", "not in the context", "not specified", "unanswerable")


def _norm(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop articles/punctuation, collapse ws."""
    s = _PUNCT.sub(" ", (s or "").lower())
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


class WikiScoringPlugin:
    """English factoid scoring (deterministic normalize→match, else defer to judge)."""

    def is_refusal(self, answer: str) -> bool:
        return any(p in (answer or "").lower() for p in _REFUSAL_EN)

    def score(self, answer: str | None, gold, case) -> ScoreResult:
        c = case if isinstance(case, dict) else case.model_dump()
        base = {"id": c["id"], "slice": c["slice"]}
        if answer is None:
            return {**base, "correct": False, "score_detail": {"error": "no answer"}}

        if c["answer_schema"] == "no_answer":
            refused = self.is_refusal(answer)
            return {**base, "correct": refused,
                    "score_detail": {"refused": refused, "over_answer": not refused}}

        # factoid: refusal on an answerable Q is wrong; else deterministic span match.
        if self.is_refusal(answer):
            return {**base, "correct": False, "score_detail": {"refused": True}}
        ans_norm = _norm(answer)
        golds = c["expected_answer"]["key_points"]
        if any(_norm(g) and _norm(g) in ans_norm for g in golds):
            return {**base, "correct": True, "score_detail": {"match": "deterministic_span"}}
        # ambiguous: no string match but the model answered → defer to judge.
        return {**base, "correct": None, "score_detail": {"deferred_to_judge": True}}


class WikiGoldMatcher:
    """Wiki gold matching at DOCUMENT granularity (doc_id), not DART's (table,file,id).

    The engine's ⊆ judgement is unchanged — it composes gold_refs ⊆ retrieved_refs
    via the shared gold_retrieved() helper (same code DART uses)."""

    def gold_refs(self, case) -> set:
        c = case if isinstance(case, dict) else case.model_dump()
        ref = c.get("source_ref")
        return {ref} if ref else set()        # no_answer cases carry a doc_id too; engine ignores empties only

    def retrieved_refs(self, entry: RunLogEntry) -> set:
        return {ch["metadata"].get("doc_id") for ch in entry.get("retrieved_chunks", [])
                if ch.get("metadata", {}).get("doc_id")}


def wiki_value_present(case, rec) -> bool:
    """Anti-illusion value-presence hook (wiki): does a gold answer string appear in
    ANY retrieved chunk? no_answer has no value → False (same convention as DART)."""
    c = case if isinstance(case, dict) else case.model_dump()
    if c["answer_schema"] != "text":
        return False
    haystack = _norm(" ".join(ch.get("text", "") for ch in rec.get("retrieved_chunks", [])))
    return any(_norm(g) and _norm(g) in haystack for g in c["expected_answer"]["key_points"])


# --------------------------------------------------------------------------- #
# G2.2 — English RAG (RAGAdapter) #2 : MiniLM embedding, doc_id metadata
# --------------------------------------------------------------------------- #

_SYSTEM_EN = (
    "Answer the question using ONLY the provided context. Keep the answer short. "
    "If the context does not contain the answer, reply exactly 'Not stated in the context.' "
    "Do not use any outside knowledge."
)


def _unique_docs(provider: WikiEvalSetProvider | None = None) -> list[dict]:
    """Distinct (doc_id, title, context) paragraphs to index as the wiki corpus."""
    path = (provider or WikiEvalSetProvider()).path
    seen, docs = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            docs.append({"doc_id": r["doc_id"], "title": r["title"], "context": r["context"]})
    return docs


def build_wiki_index(config: RagConfig = WIKI_CONFIG, rebuild: bool = False) -> int:
    """Chunk the wiki paragraphs and index them in a Chroma collection keyed by the
    config signature (MiniLM embedding) — isolated from the DART index."""
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
    for d in _unique_docs():
        for piece in splitter.split_text(d["context"]):
            docs.append(Document(page_content=piece, metadata={
                "doc_id": d["doc_id"], "title": d["title"], "source_file": f"{d['title']}.txt"}))

    persist = Path(config.persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    if rebuild and persist.exists():
        client = chromadb.PersistentClient(path=str(persist))
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)
    Chroma.from_documents(
        documents=docs, embedding=_embeddings(config),
        collection_name=name, persist_directory=str(persist),
        collection_metadata={"index_signature": config.index_signature(),
                             "embedding_model": config.embedding_model, "n_chunks": len(docs)})
    return len(docs)


class WikiRAGAdapter:
    """→ wiki Chroma index (MiniLM) + gpt-4o-mini generation. Lazy index load."""

    def __init__(self, config: RagConfig = WIKI_CONFIG):
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
                build_wiki_index(self.config)
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

        context = "\n\n".join(f"[{i}] ({c['metadata'].get('doc_id')})\n{c['text']}"
                              for i, c in enumerate(chunks, 1)) or "(no context)"
        resp = self._client.chat.completions.create(
            model=self.config.generation_model, temperature=0, seed=self.config.seed,
            messages=[{"role": "system", "content": _SYSTEM_EN},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}])
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
