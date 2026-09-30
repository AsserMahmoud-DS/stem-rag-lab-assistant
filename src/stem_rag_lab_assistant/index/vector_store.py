"""Vector store — SimpleVectorStore hydration from chunks.json. (P0.7/P1)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from stem_rag_lab_assistant.hashing import corpus_hash

logger = logging.getLogger(__name__)

CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"
_PERSIST_PATH = Path(__file__).resolve().parents[3] / "storage" / "vector_store.json"
_HASH_PATH = Path(__file__).resolve().parents[3] / "storage" / "vector_store.corpus_hash.txt"


def _load_chunks() -> dict[str, Any]:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten chunks.json into a single list of chunk dicts."""
    return [
        chunk
        for doc_data in data.get("docs", {}).values()
        for chunk in doc_data.get("chunks", [])
    ]


def _current_corpus_hash() -> str:
    """Content hash of the current corpus (chunk ids + text)."""
    return corpus_hash(_iter_chunks(_load_chunks()))


def _persisted_corpus_hash() -> str | None:
    """Read the persisted corpus-hash sentinel (None if missing/unreadable)."""
    if not _HASH_PATH.exists():
        return None
    try:
        return _HASH_PATH.read_text().strip() or None
    except OSError:
        return None


def _chunks_to_nodes(data: dict[str, Any]) -> list[TextNode]:
    """Convert every chunk in chunks.json into a LlamaIndex TextNode.

    Text is stored in node.text AND in metadata (SimpleVectorStore persists
    metadata but not text; queries return ids + scores, not nodes).
    """
    nodes: list[TextNode] = []
    for doc_data in data.get("docs", {}).values():
        for chunk in doc_data.get("chunks", []):
            meta = dict(chunk.get("metadata", {}))
            meta["_text"] = chunk["text"]
            node = TextNode(
                id_=chunk["chunk_id"],
                text=chunk["text"],
                embedding=chunk.get("embedding") or None,
                metadata=meta,
            )
            nodes.append(node)
    return nodes


def build_vector_store(
    data: dict[str, Any] | None = None,
    persist: bool = True,
) -> SimpleVectorStore:
    """Hydrate a SimpleVectorStore from chunks.json and optionally persist.

    Embeddings must already exist in chunks.json (run embed.py first).
    """
    if data is None:
        data = _load_chunks()

    nodes = _chunks_to_nodes(data)
    embedded = sum(1 for n in nodes if n.embedding and len(n.embedding) > 0)
    unembedded = len(nodes) - embedded

    if unembedded > 0:
        logger.warning(
            "%d/%d chunks have no embeddings. Run embed.py first.",
            unembedded,
            len(nodes),
        )

    store = SimpleVectorStore()
    store.add(nodes)
    logger.info("Vector store hydrated: %d nodes (%d with embeddings)", len(nodes), embedded)

    if persist:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        store.persist(str(_PERSIST_PATH))
        _HASH_PATH.write_text(corpus_hash(_iter_chunks(data)))
        logger.info("Vector store persisted to %s", _PERSIST_PATH)

    return store


def load_vector_store(persist_path: str | None = None) -> SimpleVectorStore:
    """Load a previously persisted SimpleVectorStore."""
    path = persist_path or str(_PERSIST_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(f"No persisted vector store at {path}. Run build_vector_store() first.")
    return SimpleVectorStore.from_persist_path(path)


def load_or_build_vector_store() -> SimpleVectorStore:
    """Load the persisted vector store if fresh, else rebuild from chunks.json.

    Freshness uses the corpus hash (chunk ids + text): if the corpus changed the
    store is re-hydrated from the cached embeddings (GPU-free) instead of silently
    serving a stale index.
    """
    current_hash = _current_corpus_hash()
    persisted_hash = _persisted_corpus_hash()

    if _PERSIST_PATH.exists() and persisted_hash == current_hash:
        logger.info("Loading persisted vector store (%s)", current_hash[:19])
        return load_vector_store()

    if persisted_hash is not None:
        logger.warning(
            "Vector store stale (persisted=%s, current=%s). Rebuilding from chunks.json.",
            persisted_hash[:19], current_hash[:19],
        )
    else:
        logger.info("Vector store missing or unstamped; building from chunks.json.")

    return build_vector_store()


def query_vector_store(
    store: SimpleVectorStore,
    query_embedding: list[float],
    top_k: int = 6,
) -> VectorStoreQueryResult:
    """Query the vector store with an embedding, returning top-k results.

    Returns a VectorStoreQueryResult with .nodes, .similarities, .ids.
    """
    q = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=top_k,
    )
    return store.query(q)


def retrieve(
    store: SimpleVectorStore,
    query_embedding: list[float],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """Convenience: query → list of {chunk_id, text, score, metadata} dicts.

    SimpleVectorStore.query() returns ids + similarities only (nodes=None),
    so we look up text/metadata from the store's internal metadata_dict.
    """
    result = query_vector_store(store, query_embedding, top_k)
    out: list[dict[str, Any]] = []
    ids = result.ids or []
    similarities = result.similarities or []
    metadata_dict = store._data.metadata_dict
    for chunk_id, score in zip(ids, similarities):
        meta = metadata_dict.get(chunk_id, {})
        # get (not pop): metadata_dict is the store's shared state across all
        # queries — popping _text empties it on re-retrieval within a run.
        text = meta.get("_text", "")
        out.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": float(score) if score is not None else None,
                "metadata": {k: v for k, v in meta.items() if not k.startswith("_")},
            }
        )
    return out


class VectorRetriever(BaseRetriever):
    """LlamaIndex-compatible retriever wrapping our SimpleVectorStore.

    Needed because SimpleVectorStore only persists embeddings + metadata (no
    node text), so VectorStoreIndex.from_vector_store() fails. This retriever
    uses our own retrieve() function that looks up text from metadata_dict.
    """

    def __init__(
        self,
        vector_store: SimpleVectorStore,
        embed_model: Any,
        top_k: int = 6,
    ) -> None:
        super().__init__()
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        results = retrieve(self._vector_store, query_embedding, top_k=self._top_k)

        nodes: list[NodeWithScore] = []
        for ch in results:
            node = TextNode(
                id_=ch["chunk_id"],
                text=ch["text"],
                metadata=ch.get("metadata", {}),
            )
            nodes.append(NodeWithScore(node=node, score=ch.get("score") or 0.0))
        return nodes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    store = build_vector_store()
    print(f"Nodes in store: {len(store._data.embedding_dict)}")
