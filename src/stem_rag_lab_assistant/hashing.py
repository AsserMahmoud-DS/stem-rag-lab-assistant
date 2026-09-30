"""Shared hashing helpers — one canonical content-hash scheme for the project.

The ``sha256:<hex>`` format is used wherever a file or corpus artifact is
content-addressed (OpenDataLoader parse cache, chunk ingest, BM25 freshness), so
hashes computed in different layers remain directly comparable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
