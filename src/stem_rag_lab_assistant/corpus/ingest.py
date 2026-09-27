"""Incremental ingest — build chunks.json dict-keyed by doc_id, skip on content_hash match. (P0.4)"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import (
    DATA_DIR,
    LOADED_DATA_DIR,
    get_config,
    to_relative_path,
)
from stem_rag_lab_assistant.corpus.chunking import _chunk_single_doc

logger = logging.getLogger(__name__)

CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return f"sha256:{sha.hexdigest()}"


def _load_existing_chunks() -> dict[str, Any]:
    if CHUNKS_JSON_PATH.exists():
        with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "embedding_model": get_config().embedding_model, "embedding_dim": get_config().embedding_dim, "docs": {}}


def _save_chunks(data: dict[str, Any]) -> None:
    CHUNKS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _compute_source_mtime(pdf_path: Path) -> str:
    return format(os.path.getmtime(pdf_path), ".0f")


def run_ingest(
    data_dir: Path | None = None,
    loaded_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Incremental ingest: chunk all PDFs, skip unchanged docs by content_hash.

    Returns the full chunks.json data structure.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    loaded_dir = Path(loaded_dir) if loaded_dir is not None else LOADED_DATA_DIR

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    data = _load_existing_chunks()
    docs = data.setdefault("docs", {})

    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        content_hash = _hash_file(pdf_path)
        source_mtime = _compute_source_mtime(pdf_path)

        existing = docs.get(doc_id)
        if not force and isinstance(existing, dict) and existing.get("content_hash") == content_hash:
            logger.info("  %s: unchanged (content_hash match), skipping", doc_id)
            continue

        json_path = loaded_dir / f"{doc_id}.json"
        if not json_path.exists():
            logger.warning("  %s: ODL JSON not found at %s, skipping", doc_id, json_path)
            continue

        logger.info("  %s: chunking ...", doc_id)
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        chunks = _chunk_single_doc(doc=doc, doc_path=json_path, doc_id=doc_id, pdf_path=pdf_path)

        docs[doc_id] = {
            "doc_id": doc_id,
            "file_name": doc.get("file name", f"{doc_id}.pdf"),
            "file_path": to_relative_path(pdf_path),
            "n_pages": doc.get("number of pages", 1),
            "source_mtime": source_mtime,
            "content_hash": content_hash,
            "n_chunks": len(chunks),
            "chunks": chunks,
        }
        logger.info("  %s: %d chunks produced", doc_id, len(chunks))

    _save_chunks(data)
    total_chunks = sum(d["n_chunks"] for d in docs.values())
    logger.info("Ingest complete: %d docs, %d total chunks", len(docs), total_chunks)
    return data


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Chunk dataset PDFs into chunks.json.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-chunk every document, ignoring the content_hash skip",
    )
    run_ingest(force=parser.parse_args().force)
