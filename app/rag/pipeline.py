"""Retrieve → generate RAG pipeline (Phase 1.3.2), assembled from low-level parts.

No high-level chains (no RetrievalQA): retrieval, prompt assembly, and generation
are explicit so Phase 3 can decompose where a regression occurred. Every component
is read from `RagConfig`, so changing top_k / generation_model / reranker_enabled changes
behavior. The index collection is keyed by the config signature, so we never query
a stale index built under a different config.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.env import require_openai_key

_NO_ANSWER = "정보 없음"

SYSTEM_PROMPT = (
    "주어진 컨텍스트만 근거로 한국어로 답하라. "
    f"컨텍스트에 답의 근거가 없으면 정확히 '{_NO_ANSWER}'이라고만 답하라. "
    "추측하거나 컨텍스트 밖의 지식을 사용하지 마라."
)


class Pipeline:
    def __init__(self, config: RagConfig = DEFAULT_CONFIG):
        from langchain_chroma import Chroma
        from openai import OpenAI

        from app.rag.index import _collection_exists, _embeddings

        n = _collection_exists(config)
        if n <= 0:
            raise RuntimeError(
                f"index missing for config signature {config.index_signature()} "
                f"(collection '{config.collection_name()}'). Run: python app/rag/index.py"
            )

        self.config = config
        self._embeddings = _embeddings(config)
        self._store = Chroma(
            collection_name=config.collection_name(),
            persist_directory=config.persist_dir,
            embedding_function=self._embeddings,
        )
        require_openai_key()
        self._client = OpenAI()
        self._reranker = None  # lazy; only loaded when reranker_enabled

    # --- retrieval -------------------------------------------------------- #
    def retrieve(self, question: str) -> list[dict]:
        k = self.config.top_k
        # reranker on: fetch reranker_fetch_k candidates, rerank, keep top_k.
        # (this expands the candidate pool 5→fetch_k AND reorders — not a pure
        #  reranker ablation vs baseline which fetches exactly top_k.)
        fetch = self.config.reranker_fetch_k if self.config.reranker_enabled else k
        hits = self._store.similarity_search_with_score(question, k=fetch)
        chunks = [
            {"text": doc.page_content, "metadata": dict(doc.metadata), "distance": float(score)}
            for doc, score in hits
        ]
        if self.config.reranker_enabled:
            chunks = self._rerank(question, chunks)
        return chunks[:k]

    def _rerank(self, question: str, chunks: list[dict]) -> list[dict]:
        """Reranker slot (default off). When enabled, reorder by a cross-encoder."""
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(self.config.reranker_model, max_length=512)
        scores = self._reranker.predict([(question, c["text"]) for c in chunks])
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    # --- generation ------------------------------------------------------- #
    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        blocks = []
        for i, c in enumerate(chunks, 1):
            m = c["metadata"]
            tag = m.get("table_id") or m.get("section") or f"p.{m.get('page')}"
            blocks.append(f"[{i}] ({m.get('company')} {m.get('fiscal_year')} · {m.get('source_file')} · {tag})\n{c['text']}")
        context = "\n\n".join(blocks) if blocks else "(컨텍스트 없음)"
        return f"컨텍스트:\n{context}\n\n질문: {question}\n답:"

    def generate(self, question: str, chunks: list[dict]) -> dict:
        user_prompt = self._build_prompt(question, chunks)
        full_prompt = f"[system]\n{SYSTEM_PROMPT}\n\n[user]\n{user_prompt}"

        if not chunks:  # empty context -> refuse without an LLM call
            return {"answer": _NO_ANSWER, "prompt": full_prompt, "llm_calls": 0,
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

        resp = self._client.chat.completions.create(
            model=self.config.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            seed=self.config.seed,  # determinism hygiene (residual wobble = noise band)
        )
        u = resp.usage
        return {
            "answer": (resp.choices[0].message.content or "").strip(),
            "prompt": full_prompt,
            "llm_calls": 1,
            "token_usage": {"prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens},
        }

    # --- end to end ------------------------------------------------------- #
    def run(self, question: str) -> dict:
        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        gen = self.generate(question, chunks)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "question": question,
            "retrieved_chunks": chunks,
            "prompt": gen["prompt"],
            "answer": gen["answer"],
            "latency_ms": latency_ms,
            "llm_calls": gen["llm_calls"],
            "token_usage": gen["token_usage"],
        }


def _demo() -> None:
    """Run 3 sample questions (table_value / comparison / no_answer)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    cases = [json.loads(l) for l in Path("data/eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    samples = {
        "table_value": next(c for c in cases if c["slice"] == "table_value"),
        "comparison": next(c for c in cases if c["answer_schema"] == "comparison"),
        "no_answer": next(c for c in cases if c["slice"] == "no_answer"),
    }
    pipe = Pipeline()
    for label, case in samples.items():
        r = pipe.run(case["question"])
        print("\n" + "=" * 80)
        print(f"[{label}] {case['id']}")
        print("Q:", r["question"])
        print(f"retrieved {len(r['retrieved_chunks'])} chunks (company/year/tag):")
        for i, c in enumerate(r["retrieved_chunks"], 1):
            m = c["metadata"]
            print(f"  [{i}] {m.get('company')} {m.get('fiscal_year')} · {m.get('table_id') or m.get('section')} · is_table={m.get('is_table')} (dist {c['distance']:.3f})")
        print("answer:", r["answer"])
        print(f"latency_ms={r['latency_ms']}  llm_calls={r['llm_calls']}  tokens={r['token_usage']}")


if __name__ == "__main__":
    _demo()
