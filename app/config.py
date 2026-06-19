"""Single source of truth for the RAG pipeline configuration (Phase 1.3).

Every swappable component lives here so Phase 5 can inject regressions by changing
one field. All runs must log the full config (see `RagConfig.fingerprint`).

Index consistency: the vector index depends only on chunk_size, chunk_overlap and
embedding_model. `index_signature` hashes exactly those, and the Chroma collection
name embeds that signature — so changing any of them targets a *different*
collection and you can never search a stale index built with a different config.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


class RagConfig(BaseModel):
    # --- swap targets (Phase 5 regression levers) ---
    chunk_size: int = 1000          # chars (non-table prose)
    chunk_overlap: int = 150
    top_k: int = 5
    embedding_model: str = "BAAI/bge-m3"
    generation_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"  # different model than generation (avoid self-bias)
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # used only when enabled (1.3.2)
    reranker_fetch_k: int = 20  # when reranker on: fetch this many, then rerank → top_k

    # --- auxiliary (not regression targets) ---
    vectorstore: str = "chroma"
    persist_dir: str = "data/index"
    collection_base: str = "dart"
    seed: int = 0

    def index_signature(self) -> str:
        """Hash of fields that determine index content (12 hex chars)."""
        payload = json.dumps(
            {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def collection_name(self) -> str:
        """Collection name keyed by index signature (stale-index-proof)."""
        return f"{self.collection_base}_{self.index_signature()}"

    def fingerprint(self) -> dict:
        """Full, log-friendly snapshot (+ derived signature/collection)."""
        return {
            **self.model_dump(),
            "index_signature": self.index_signature(),
            "collection_name": self.collection_name(),
        }


DEFAULT_CONFIG = RagConfig()
