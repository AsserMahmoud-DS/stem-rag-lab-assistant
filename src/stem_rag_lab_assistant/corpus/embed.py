"""Embed — bge-m3 dense embedding cache, idempotent, hydrates chunks.json embeddings. (P0.6)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from stem_rag_lab_assistant.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"


def _load_chunks() -> dict[str, Any]:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunks(data: dict[str, Any]) -> None:
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_embed(batch_size: int = 16) -> dict[str, Any]:
    """Embed all unembedded chunks in chunks.json using bge-m3.

    Idempotent — chunks that already have an embedding vector of the correct
    dimensionality are skipped.  Chunks with an empty [] embedding are re-embedded.
    Modifies chunks.json in place.
    """
    if not CHUNKS_JSON_PATH.exists():
        raise FileNotFoundError(f"chunks.json not found at {CHUNKS_JSON_PATH}. Run ingest first.")

    data = _load_chunks()

    # Collect all chunks that need embedding
    todo: list[tuple[str, int, str]] = []  # (doc_id, chunk_idx, text)
    for doc_id, doc_data in data.get("docs", {}).items():
        for idx, chunk in enumerate(doc_data.get("chunks", [])):
            emb = chunk.get("embedding")
            if not emb or len(emb) != EMBEDDING_DIM:
                todo.append((doc_id, idx, chunk["text"]))

    if not todo:
        logger.info("All chunks already embedded. Nothing to do.")
        return data

    logger.info("Embedding %d chunks with %s (dim=%d)", len(todo), EMBEDDING_MODEL_NAME, EMBEDDING_DIM)

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        embed_batch_size=batch_size,
        device = "cuda"
    )

    texts = [t for _, _, t in todo]
    embeddings = embed_model.get_text_embedding_batch(texts)

    for (doc_id, idx, _), emb in zip(todo, embeddings):
        data["docs"][doc_id]["chunks"][idx]["embedding"] = emb

    _save_chunks(data)
    logger.info("Embedding complete: %d chunks embedded", len(todo))
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_embed()
