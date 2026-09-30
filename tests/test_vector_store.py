"""Regression tests for the vector store — shared-state safety of retrieve()."""

from __future__ import annotations

import json

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore

from stem_rag_lab_assistant.index import vector_store as vs
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


def _write_chunks(path, pairs, dim: int = 3) -> None:
    chunks = [
        {
            "chunk_id": cid,
            "text": text,
            "embedding": [1.0 if j == i else 0.0 for j in range(dim)],
            "metadata": {},
        }
        for i, (cid, text) in enumerate(pairs)
    ]
    data = {
        "version": 1,
        "embedding_model": "x",
        "embedding_dim": dim,
        "docs": {"d": {"doc_id": "d", "n_chunks": len(chunks), "chunks": chunks}},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_or_build_builds_then_reuses_then_rebuilds(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.json"
    persist_path = tmp_path / "storage" / "vector_store.json"
    hash_path = tmp_path / "storage" / "vector_store.corpus_hash.txt"
    monkeypatch.setattr(vs, "CHUNKS_JSON_PATH", chunks_path)
    monkeypatch.setattr(vs, "_PERSIST_PATH", persist_path)
    monkeypatch.setattr(vs, "_HASH_PATH", hash_path)

    _write_chunks(chunks_path, [("d::ch_1", "alpha"), ("d::ch_2", "beta")])
    vs.load_or_build_vector_store()

    assert persist_path.exists()
    assert hash_path.exists()
    hash_1 = hash_path.read_text()

    # Unchanged corpus -> persisted store reused, sentinel untouched.
    vs.load_or_build_vector_store()
    assert hash_path.read_text() == hash_1

    # Changed chunk text -> rebuild -> new hash.
    _write_chunks(chunks_path, [("d::ch_1", "alpha"), ("d::ch_2", "GAMMA")])
    vs.load_or_build_vector_store()
    assert hash_path.read_text() != hash_1
