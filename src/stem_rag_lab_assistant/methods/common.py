"""Common post-processor — attach_images() + shared context formatting for all methods."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import get_config

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


# ---------------------------------------------------------------------------
# Shared retrieval-context formatting (matched context across methods)
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Avoids importing a tokenizer."""
    return max(1, len(text) // 4)


def budget_graph_text(
    entities: list[tuple[str, str]],
    relations: list[tuple[str, str, str]],
) -> str:
    """Render token-budgeted ENTITIES / RELATIONS blocks (LightRAG-style).

    Entity/relation descriptions are inserted into the answer context so every
    method sees the same *shape* of context. Caps are the ``lh_max_*`` ceilings
    (entity/relation match native LightRAG's 6000/8000 defaults; the combined
    ``lh_max_graph_tokens`` is an extra safety net that stays inert while the
    per-part caps sum to it).
    """
    cfg = get_config()

    ent_lines: list[str] = []
    used_e = 0
    for name, desc in entities:
        line = f"- {name}: {desc}"
        t = approx_tokens(line)
        if used_e + t > cfg.lh_max_entity_tokens:
            break
        ent_lines.append(line)
        used_e += t

    rel_lines: list[str] = []
    used_r = 0
    for src, dst, desc in relations:
        line = f"- {src} -> {dst}: {desc}"
        t = approx_tokens(line)
        if used_r + t > cfg.lh_max_relation_tokens:
            break
        rel_lines.append(line)
        used_r += t

    while (used_e + used_r) > cfg.lh_max_graph_tokens and (ent_lines or rel_lines):
        if rel_lines:
            used_r -= approx_tokens(rel_lines.pop())
        elif ent_lines:
            used_e -= approx_tokens(ent_lines.pop())

    blocks: list[str] = []
    if ent_lines:
        blocks.append("ENTITIES:\n" + "\n".join(ent_lines))
    if rel_lines:
        blocks.append("RELATIONS:\n" + "\n".join(rel_lines))
    return "\n\n".join(blocks)


def format_retrieval_context(
    chunks: list[dict[str, Any]],
    graph_text: str = "",
) -> str:
    """Format the shared answer context: optional graph text + CHUNKS block."""
    chunk_block = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)
    parts: list[str] = []
    if graph_text:
        parts.append(graph_text)
    parts.append("CHUNKS:\n" + chunk_block)
    return "\n\n".join(parts)
