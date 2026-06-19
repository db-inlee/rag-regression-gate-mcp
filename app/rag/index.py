"""Build a local Chroma index from the extracted corpus (Phase 1.3.1).

The collection name embeds the config index-signature (chunk_size, chunk_overlap,
embedding_model), so a config change targets a different collection — searching a
stale index built under a different config is impossible. The full config
fingerprint is stored both in the collection metadata and a sidecar .meta.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import DEFAULT_CONFIG, RagConfig
from app.rag.chunker import Chunk, chunk_corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("index")

EXTRACTED_DIR = Path("data/corpus/extracted")


def _embeddings(config: RagConfig):
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
    )
    # bge-m3 defaults to 8192 tokens; long garbled wide-table rows then blow up
    # MPS attention memory (O(seq^2)). Cap to 512 tokens — chunks are short and
    # the year header is at the start of each chunk, so it survives truncation.
    client = getattr(emb, "_client", None) or getattr(emb, "client", None)
    if client is not None:
        try:
            client.max_seq_length = 512
        except Exception:  # noqa: BLE001
            pass
    return emb


def _collection_exists(config: RagConfig) -> int:
    """Return chunk count of the existing collection for this config, else -1."""
    import chromadb

    persist = Path(config.persist_dir)
    if not persist.exists():
        return -1
    client = chromadb.PersistentClient(path=str(persist))
    name = config.collection_name()
    if name not in [c.name for c in client.list_collections()]:
        return -1
    return client.get_collection(name).count()


def _log_distribution(chunks: list[Chunk]) -> None:
    by_company = Counter(c.metadata["company"] for c in chunks)
    by_year = Counter(c.metadata["fiscal_year"] for c in chunks)
    n_table = sum(1 for c in chunks if c.metadata["is_table"])
    logger.info("total chunks: %d  (table %d / prose %d)", len(chunks), n_table, len(chunks) - n_table)
    logger.info("  by company: %s", dict(by_company))
    logger.info("  by year   : %s", dict(sorted(by_year.items())))


def build_index(config: RagConfig, rebuild: bool = False) -> int:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    logger.info("config fingerprint: %s", json.dumps(config.fingerprint(), ensure_ascii=False))
    name = config.collection_name()

    existing = _collection_exists(config)
    if existing > 0 and not rebuild:
        logger.info("index up-to-date: collection '%s' has %d chunks (use --rebuild to force)",
                    name, existing)
        return existing

    chunks = chunk_corpus(EXTRACTED_DIR, config)
    if not chunks:
        logger.error("no chunks produced from %s", EXTRACTED_DIR)
        return 0
    _log_distribution(chunks)

    docs = [Document(page_content=c.text, metadata=c.metadata) for c in chunks]
    embeddings = _embeddings(config)

    persist = Path(config.persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    if rebuild:
        import chromadb
        client = chromadb.PersistentClient(path=str(persist))
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)
            logger.info("deleted existing collection '%s' for rebuild", name)

    logger.info("embedding %d chunks with %s -> collection '%s'",
                len(docs), config.embedding_model, name)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=name,
        persist_directory=str(persist),
        collection_metadata={
            "index_signature": config.index_signature(),
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "n_chunks": len(docs),
        },
    )

    sidecar = persist / f"{name}.meta.json"
    sidecar.write_text(
        json.dumps({**config.fingerprint(), "n_chunks": len(docs)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote index (%d chunks) + sidecar %s", len(docs), sidecar)
    return len(docs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if collection exists")
    args = ap.parse_args()
    build_index(DEFAULT_CONFIG, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
