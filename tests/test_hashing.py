"""Tests for the shared hashing helper."""

from __future__ import annotations

import hashlib

from stem_rag_lab_assistant.hashing import corpus_hash, dataset_hash, file_sha256


def test_file_sha256_matches_hashlib(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert file_sha256(path) == "sha256:" + hashlib.sha256(b"hello world").hexdigest()


def test_file_sha256_streams_large_file(tmp_path) -> None:
    data = b"x" * 200_000  # spans several read blocks
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert file_sha256(path) == "sha256:" + hashlib.sha256(data).hexdigest()


def test_file_sha256_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert file_sha256(path) == "sha256:" + hashlib.sha256(b"").hexdigest()


def _chunks(pairs):
    return [{"chunk_id": cid, "text": text} for cid, text in pairs]


def test_corpus_hash_is_order_independent() -> None:
    a = _chunks([("d::ch_1", "alpha"), ("d::ch_2", "beta")])
    b = list(reversed(a))
    assert corpus_hash(a) == corpus_hash(b)


def test_corpus_hash_changes_on_text_change() -> None:
    a = _chunks([("d::ch_1", "alpha")])
    b = _chunks([("d::ch_1", "ALPHA")])
    assert corpus_hash(a) != corpus_hash(b)


def test_corpus_hash_changes_on_add_and_remove() -> None:
    a = _chunks([("d::ch_1", "alpha")])
    added = _chunks([("d::ch_1", "alpha"), ("d::ch_2", "beta")])
    assert corpus_hash(a) != corpus_hash(added)


def test_corpus_hash_is_prefixed() -> None:
    assert corpus_hash(_chunks([("d::ch_1", "alpha")])).startswith("sha256:")


def _questions(rows):
    return [
        {"question_id": qid, "question": q, "golden_answer": a, "category": c}
        for qid, q, a, c in rows
    ]


def test_dataset_hash_is_order_independent() -> None:
    a = _questions([("Q1", "q1", "a1", "lookup"), ("Q2", "q2", "a2", "multi_hop")])
    assert dataset_hash(a) == dataset_hash(list(reversed(a)))


def test_dataset_hash_changes_on_edit() -> None:
    a = _questions([("Q1", "q1", "a1", "lookup")])
    b = _questions([("Q1", "q1", "a1 edited", "lookup")])
    assert dataset_hash(a) != dataset_hash(b)


def test_dataset_hash_ignores_meta_fields() -> None:
    base = _questions([("Q1", "q1", "a1", "lookup")])
    with_meta = [dict(base[0], source_chunk_ids=["d::ch_1"], doc_ids_involved=["d"])]
    assert dataset_hash(base) == dataset_hash(with_meta)
