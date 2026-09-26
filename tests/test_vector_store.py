"""Regression tests for the vector store — shared-state safety of retrieve()."""

from __future__ import annotations

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore

from stem_rag_lab_assistant.index.vector_store import retrieve


def _make_store() -> SimpleVectorStore:
    """Minimal 3-chunk store mirroring chunks.json hydration (with `_text`)."""
    store = SimpleVectorStore()
    texts = ["alpha resistor text", "beta capacitor text", "gamma inductor text"]
    nodes = [
        TextNode(
            id_=f"doc::ch_{i:05d}",
            text=text,
            embedding=[1.0 if j == i else 0.0 for j in range(3)],
            metadata={"chunk_order_index": i, "_text": text},
        )
        for i, text in enumerate(texts)
    ]
    store.add(nodes)
    return store


def test_retrieve_repeated_calls_keep_chunk_text() -> None:
    """S0 regression: retrieve() must not mutate the shared store.

    Previously it did ``meta.pop("_text")``, so a second retrieval of the same
    chunk returned empty text — silently degrading naive/hybrid/lightrag_hybrid
    across a multi-question run (but not the LightRAG baseline).
    """
    store = _make_store()
    query_embedding = [1.0, 0.0, 0.0]

    first = retrieve(store, query_embedding, top_k=3)
    second = retrieve(store, query_embedding, top_k=3)

    assert [c["text"] for c in first] == [c["text"] for c in second]
    assert all(c["text"] for c in second), "chunk text was emptied on re-retrieval"


def test_retrieve_leaves_private_text_in_store_metadata() -> None:
    """`_text` stays in store.metadata_dict but never leaks into returned metadata."""
    store = _make_store()
    results = retrieve(store, [1.0, 0.0, 0.0], top_k=3)

    for meta in store._data.metadata_dict.values():
        assert "_text" in meta

    for chunk in results:
        assert "_text" not in chunk["metadata"]
