"""Tests for the shared hashing helper."""

from __future__ import annotations

import hashlib

from stem_rag_lab_assistant.hashing import file_sha256


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
