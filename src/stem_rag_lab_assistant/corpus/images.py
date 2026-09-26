"""Images — build images.json from ODL image + caption + ai_description data, back-fill linked_image_ids. (P0.5)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import LOADED_DATA_DIR, to_relative_path

logger = logging.getLogger(__name__)

IMAGES_JSON_PATH = Path(__file__).resolve().parents[3] / "images.json"
CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"


def _walk_images(
    kids: list[dict[str, Any]],
    images: list[dict[str, Any]],
    doc_id: str,
    loaded_dir: Path,
    img_counter: dict[str, int],
) -> None:
    """In-order walk of ODL element tree, collecting image metadata.

    Uses the SAME deterministic sequential counter as chunking._walk_elements
    to ensure image_ids match those stored in linked_image_ids on chunks.
    """
    for element in kids:
        if not isinstance(element, dict):
            continue

        elem_type = element.get("type", "")

        page_raw = element.get("page number")
        page = int(page_raw) if page_raw is not None else None
        bbox = element.get("bounding box")

        if elem_type == "image":
            img_id = f"{doc_id}::img_{img_counter['count']:05d}"
            img_counter["count"] += 1

            source = element.get("source", "")
            source_path = to_relative_path(loaded_dir / source) if source else ""
            fmt = element.get("format") or _infer_format(source) or "png"

            # ai_description comes from "description" (older) or "alt" (newer)
            ai_description = element.get("description") or element.get("alt") or ""

            images.append(
                {
                    "image_id": img_id,
                    "doc_id": doc_id,
                    "page": page or 0,
                    "bbox": bbox,
                    "source_path": source_path,
                    "caption_text": "",
                    "ai_description": ai_description,
                    "linked_chunk_id": None,
                    "format": fmt,
                }
            )

        elif elem_type == "caption":
            linked_id = element.get("linked content id")
            text = element.get("content", "")
            # Captions are rare (0 in our corpus), but record when present
            if linked_id is not None:
                logger.debug("  caption w/ linked_id=%d: %s", linked_id, text[:80])

        for key in ("kids", "list items", "rows", "cells"):
            children = element.get(key)
            if isinstance(children, list):
                _walk_images(children, images, doc_id, loaded_dir, img_counter)


def _infer_format(source: str) -> str:
    src = source.lower()
    if src.endswith(".png"):
        return "png"
    if src.endswith(".jpg") or src.endswith(".jpeg"):
        return "jpeg"
    return "png"


def _load_chunks() -> dict[str, Any]:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunks(data: dict[str, Any]) -> None:
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_images(
    loaded_dir: Path | None = None,
) -> dict[str, Any]:
    """Build images.json from loaded_data ODL JSONs.

    Walks each doc's ODL element tree to collect image metadata (using the same
    sequential ID scheme as the chunker), then cross-references with chunks.json
    to set linked_chunk_id.  Back-fills chunk.metadata.linked_image_ids for
    consistency.
    """
    loaded_dir = Path(loaded_dir) if loaded_dir is not None else LOADED_DATA_DIR
    chunks_data = _load_chunks()

    all_images: list[dict[str, Any]] = []

    json_files = sorted(loaded_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No ODL JSONs found in {loaded_dir}")

    for jf in json_files:
        doc_id = jf.stem
        with open(jf, "r", encoding="utf-8") as f:
            doc = json.load(f)

        img_counter: dict[str, int] = {"count": 0}
        doc_images: list[dict[str, Any]] = []
        _walk_images(doc.get("kids", []), doc_images, doc_id, loaded_dir, img_counter)

        # Cross-reference: for each image, find which chunk links to it
        doc_chunks = (
            chunks_data.get("docs", {}).get(doc_id, {}).get("chunks", [])
        )
        img_to_chunk: dict[str, str] = {}
        for chunk in doc_chunks:
            for linked_id in chunk.get("metadata", {}).get("linked_image_ids", []):
                img_to_chunk[linked_id] = chunk["chunk_id"]

        for img in doc_images:
            chunk_id = img_to_chunk.get(img["image_id"])
            if chunk_id:
                img["linked_chunk_id"] = chunk_id

        all_images.extend(doc_images)
        logger.info(
            "  %s: %d images (%d linked to a chunk)",
            doc_id,
            len(doc_images),
            sum(1 for img in doc_images if img["linked_chunk_id"]),
        )

    images_data: dict[str, Any] = {"version": 1, "images": all_images}

    with open(IMAGES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(images_data, f, indent=2, ensure_ascii=False)

    logger.info("images.json written: %d total images", len(all_images))
    return images_data


def _backfill_chunk_image_ids(chunks_data: dict[str, Any], images_data: dict[str, Any]) -> int:
    """Back-fill chunk.metadata.linked_image_ids from images.json.linked_chunk_id.

    Returns number of chunks updated. This is a reconciliation step — the chunker
    normally sets these, but running this ensures images.json is the source of truth
    and chunks.json stays consistent.
    """
    img_to_chunks: dict[str, list[str]] = {}
    for img in images_data.get("images", []):
        img_id = img["image_id"]
        chunk_id = img.get("linked_chunk_id")
        if chunk_id:
            img_to_chunks.setdefault(img_id, []).append(chunk_id)

    updated = 0
    for doc_data in chunks_data.get("docs", {}).values():
        for chunk in doc_data.get("chunks", []):
            existing = set(chunk["metadata"].get("linked_image_ids", []))
            new_ids = [
                img_id
                for img_id, chunk_ids in img_to_chunks.items()
                if chunk["chunk_id"] in chunk_ids and img_id not in existing
            ]
            if new_ids:
                chunk["metadata"]["linked_image_ids"] = sorted(existing | set(new_ids))
                updated += 1

    if updated:
        _save_chunks(chunks_data)
        logger.info("Back-filled linked_image_ids on %d chunks", updated)
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    images_data = run_images()
    # Back-fill for consistency
    chunks_data = _load_chunks()
    _backfill_chunk_image_ids(chunks_data, images_data)
