"""BM25 store — rank_bm25 index over chunk texts with disk persistence. (P2.1)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.hashing import corpus_hash

logger = logging.getLogger(__name__)

CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"
_PERSIST_DIR = Path(__file__).resolve().parents[3] / "storage" / "bm25_store"

_BM25_RETRIEVER: BM25Retriever | None = None


def _load_all_chunks() -> list[dict[str, Any]]:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for doc_data in data.get("docs", {}).values():
        chunks.extend(doc_data.get("chunks", []))
    return chunks


def _current_corpus_hash() -> str:
    """Content hash of the current corpus (chunk ids + text)."""
    return corpus_hash(_load_all_chunks())


def _chunks_to_nodes(chunks: list[dict[str, Any]]) -> list[TextNode]:
    nodes = []
    for ch in chunks:
        meta = dict(ch.get("metadata", {}))
        node = TextNode(
            id_=ch["chunk_id"],
            text=ch["text"],
            metadata=meta,
        )
        nodes.append(node)
    return nodes


def _persisted_corpus_hash() -> str | None:
    """Read the persisted corpus-hash sentinel (None if missing/unreadable)."""
    sentinel = _PERSIST_DIR / "corpus_hash.txt"
    if not sentinel.exists():
        return None
    try:
        return sentinel.read_text().strip() or None
    except OSError:
        return None


def get_bm25_retriever() -> BM25Retriever:
    """Lazy-init singleton: load persisted BM25 index if fresh, else build + persist.

    Freshness is decided by a corpus hash over the chunk ids + text, so adding,
    removing, or editing chunks forces a rebuild (a global-statistic index like
    BM25 cannot be updated incrementally).
    """
    global _BM25_RETRIEVER
    if _BM25_RETRIEVER is not None:
        return _BM25_RETRIEVER

    current_hash = _current_corpus_hash()
    persisted_hash = _persisted_corpus_hash()

    if persisted_hash == current_hash:
        logger.info("Loading persisted BM25 index (%s) from %s", current_hash[:19], _PERSIST_DIR)
        _BM25_RETRIEVER = BM25Retriever.from_persist_dir(str(_PERSIST_DIR))
    else:
        if persisted_hash is not None:
            logger.info(
                "BM25 cache stale (persisted=%s, current=%s). Rebuilding.",
                persisted_hash[:19], current_hash[:19],
            )
        chunks = _load_all_chunks()
        nodes = _chunks_to_nodes(chunks)
        logger.info("Building BM25 index over %d chunks", len(nodes))

        _BM25_RETRIEVER = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=get_config().bm25_top_k,
        )
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _BM25_RETRIEVER.persist(str(_PERSIST_DIR))
        (_PERSIST_DIR / "corpus_hash.txt").write_text(current_hash)
        logger.info("BM25 retriever persisted to %s (top_k=%d)", _PERSIST_DIR, get_config().bm25_top_k)

    return _BM25_RETRIEVER


def get_bm25_retriever_at(top_k: int) -> BM25Retriever:
    """Private BM25Retriever instance with its own ``similarity_top_k``.

    Loads the same persisted index as the shared singleton (calling it first
    also guarantees the persisted index exists and is fresh), so a caller can
    widen top-k for its own fusion retriever without mutating state other
    methods see through the shared singleton.
    """
    get_bm25_retriever()
    retriever = BM25Retriever.from_persist_dir(str(_PERSIST_DIR))
    retriever.similarity_top_k = top_k
    return retriever


def bm25_retrieve(
    query: str,
    retriever: BM25Retriever | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Keyword (BM25) retrieval — return [{chunk_id, text, score, metadata}]."""
    bm25 = retriever or get_bm25_retriever()

    if top_k is not None and top_k != bm25.similarity_top_k:
        bm25.similarity_top_k = top_k

    nodes_with_scores = bm25.retrieve(query)

    results: list[dict[str, Any]] = []
    for nws in nodes_with_scores:
        results.append({
            "chunk_id": nws.node.node_id,
            "text": nws.node.text or "",
            "score": float(nws.score) if nws.score is not None else None,
            "metadata": dict(nws.node.metadata or {}),
        })

    logger.info("BM25 retrieval: %d results for query: %r", len(results), query[:60])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = bm25_retrieve("What is a Wheatstone bridge?")
    for r in results:
        print(f"{r['chunk_id']}: {r['score']:.4f} — {r['text'][:100]}")
