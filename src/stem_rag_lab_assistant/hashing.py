"""Shared hashing helpers — one canonical content-hash scheme for the project.

The ``sha256:<hex>`` format is used wherever a file or corpus artifact is
content-addressed (OpenDataLoader parse cache, chunk ingest, BM25 freshness), so
hashes computed in different layers remain directly comparable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 65536


def file_sha256(path: Path) -> str:
    """Return the content hash of a file as ``sha256:<hex>``.

    Reads in blocks so it stays memory-safe for large PDFs/artifacts.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK_SIZE), b""):
            sha.update(block)
    return f"sha256:{sha.hexdigest()}"


def corpus_hash(chunks: Iterable[Mapping[str, Any]]) -> str:
    """Order-independent content hash of a collection of chunks as ``sha256:<hex>``.

    Computed over the sorted ``(chunk_id, sha256(text))`` pairs, so it changes
    whenever a chunk is added, removed, or its text changes, while remaining
    stable across chunk ordering. This is the canonical corpus fingerprint used
    for cache freshness and run provenance.
    """
    digest = hashlib.sha256()
    entries = sorted(
        (
            str(chunk["chunk_id"]),
            hashlib.sha256(str(chunk.get("text", "")).encode("utf-8")).hexdigest(),
        )
        for chunk in chunks
    )
    for chunk_id, text_digest in entries:
        digest.update(f"{chunk_id}\x00{text_digest}\n".encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def dataset_hash(questions: Iterable[Mapping[str, Any]]) -> str:
    """Order-independent content hash of a question set as ``sha256:<hex>``.

    Hashes ``question_id`` + ``question`` + ``golden_answer`` + ``category`` per
    question, sorted by ``question_id``, so it changes when a question is added,
    removed, or edited — independent of question ordering and of provenance
    fields (e.g. ``source_chunk_ids``, meta) that are not part of the content.
    """
    digest = hashlib.sha256()
    entries = sorted(
        (
            str(q.get("question_id", "")),
            str(q.get("question", "")),
            str(q.get("golden_answer", "")),
            str(q.get("category", "")),
        )
        for q in questions
    )
    for parts in entries:
        digest.update(("\x00".join(parts) + "\n").encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"
