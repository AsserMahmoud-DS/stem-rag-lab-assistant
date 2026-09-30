"""Tests for BM25 corpus-hash freshness (rebuild vs. reuse persisted index)."""

from __future__ import annotations

import json

from stem_rag_lab_assistant.index import bm25_store


def _write_chunks(path, pairs) -> None:
    data = {
        "version": 1,
        "embedding_model": "x",
        "embedding_dim": 4,
        "docs": {
            "d": {
                "doc_id": "d",
                "n_chunks": len(pairs),
                "chunks": [
                    {"chunk_id": cid, "text": text, "metadata": {}}
                    for cid, text in pairs
                ],
            }
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _fresh(monkeypatch) -> None:
    monkeypatch.setattr(bm25_store, "_BM25_RETRIEVER", None)


def test_bm25_rebuilds_only_when_corpus_changes(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.json"
    persist_dir = tmp_path / "bm25"
    monkeypatch.setattr(bm25_store, "CHUNKS_JSON_PATH", chunks_path)
    monkeypatch.setattr(bm25_store, "_PERSIST_DIR", persist_dir)

    _write_chunks(chunks_path, [("d::ch_1", "wheatstone bridge balance"),
                                ("d::ch_2", "oscilloscope lissajous")])
    _fresh(monkeypatch)
    bm25_store.get_bm25_retriever()

    sentinel = persist_dir / "corpus_hash.txt"
    assert sentinel.exists()
    hash_1 = sentinel.read_text()

    # Unchanged corpus -> persisted index reused, sentinel untouched.
    _fresh(monkeypatch)
    bm25_store.get_bm25_retriever()
    assert sentinel.read_text() == hash_1

    # Edited chunk text -> rebuild -> new hash.
    _write_chunks(chunks_path, [("d::ch_1", "wheatstone bridge BALANCE"),
                                ("d::ch_2", "oscilloscope lissajous")])
    _fresh(monkeypatch)
    bm25_store.get_bm25_retriever()
    hash_2 = sentinel.read_text()
    assert hash_2 != hash_1

    # Added chunk -> rebuild -> new hash again.
    _write_chunks(chunks_path, [("d::ch_1", "wheatstone bridge BALANCE"),
                                ("d::ch_2", "oscilloscope lissajous"),
                                ("d::ch_3", "new chunk")])
    _fresh(monkeypatch)
    bm25_store.get_bm25_retriever()
    assert sentinel.read_text() != hash_2
