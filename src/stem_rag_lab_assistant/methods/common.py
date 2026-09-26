"""Common post-processor — attach_images() for all 4 methods (reads images.json)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IMAGES_JSON_PATH = Path(__file__).resolve().parents[3] / "images.json"


def _load_images(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _IMAGES_JSON_PATH
    if not p.exists():
        logger.warning("images.json not found at %s", p)
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("images", [])


def attach_images(
    retrieved_chunks: list[dict[str, Any]],
    images_json_path: Path | None = None,
    max_images: int = 3,
) -> list[dict[str, Any]]:
    """For each retrieved chunk, look up linked images in images.json.

    Sorts image entries by their host chunk's retrieval score (descending),
    then takes the top ``max_images``. This ensures the most relevant
    images (whose caption/description chunk ranked highest) come first.

    Args:
        retrieved_chunks: list of {chunk_id, text, score, metadata} dicts
            (score may be None for graph-expanded chunks).
        images_json_path: override path to images.json.
        max_images: cap the number of returned image refs.

    Returns:
        List of image ref dicts with keys:
        image_id, source_path, caption_text, ai_description,
        linked_chunk_id, chunk_score.
    """
    images = _load_images(images_json_path)
    if not images:
        return []

    # Build lookup: linked_chunk_id → list of image entries
    linked: dict[str, list[dict[str, Any]]] = {}
    for img in images:
        cid = img.get("linked_chunk_id")
        if cid:
            linked.setdefault(cid, []).append(img)

    # Build score lookup from retrieved chunks
    chunk_scores: dict[str, float] = {}
    for ch in retrieved_chunks:
        score = ch.get("score")
        chunk_scores[ch["chunk_id"]] = float(score) if score is not None else 0.0

    # Collect image refs with their host chunk score
    scored: list[dict[str, Any]] = []
    for ch in retrieved_chunks:
        cid = ch["chunk_id"]
        for img in linked.get(cid, []):
            scored.append(
                {
                    "image_id": img["image_id"],
                    "source_path": img.get("source_path", ""),
                    "caption_text": img.get("caption_text", ""),
                    "ai_description": img.get("ai_description", ""),
                    "linked_chunk_id": cid,
                    "chunk_score": chunk_scores.get(cid, 0.0),
                }
            )

    # Sort by chunk score descending, cap
    scored.sort(key=lambda x: x["chunk_score"], reverse=True)
    scored = scored[:max_images]

    if scored:
        logger.debug(
            "attach_images: %d image refs returned (cap=%d) from %d retrieved chunks",
            len(scored), max_images, len(retrieved_chunks),
        )

    return scored
