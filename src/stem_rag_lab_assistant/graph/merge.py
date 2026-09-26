"""Graph merge — entity normalization (case-insensitive) + relation dedup. (P4)"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def _entity_id(normalized_name: str) -> str:
    """Stable entity_id from normalized name: ``e_{sha256[:8]}``."""
    h = hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()[:8]
    return f"e_{h}"


def _relation_id(src_id: str, dst_id: str, rel_type: str) -> str:
    """Stable relation_id from (src, dst, type): ``r_{sha256[:8]}``."""
    key = f"{src_id}|{dst_id}|{rel_type}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"r_{h}"


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name: strip, collapse whitespace, title case."""
    if not name or not name.strip():
        return ""
    return " ".join(name.strip().split()).title()


def merge_entities(
    raw_extractions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge entities across raw extractions into a normalized entity registry.

    Strategy:
      - Normalize entity names (case-insensitive, whitespace).
      - Group extractions by normalized name.
      - Entity type: majority vote across chunks.
      - Descriptions: concatenated unique descriptions (\" | \" separated).
      - source_chunk_ids: union across all occurrences.
      - entity_id: stable hash of the normalized name.

    Returns dict keyed by normalized entity name:
        {normalized_name: {entity_id, name, type, description, source_chunk_ids: set}}
    """
    groups: dict[str, dict[str, Any]] = {}

    for raw in raw_extractions:
        chunk_id = raw["chunk_id"]
        for ent in raw.get("entities", []):
            raw_name = ent.get("name", "")
            normalized = normalize_entity_name(raw_name)
            if not normalized:
                continue

            if normalized not in groups:
                groups[normalized] = {
                    "entity_id": _entity_id(normalized),
                    "name": normalized,
                    "type_counts": Counter(),
                    "descriptions": [],
                    "source_chunk_ids": set(),
                }

            entry = groups[normalized]
            ent_type = ent.get("type", "Other")
            if not ent_type or ent_type == "?":
                ent_type = "Other"
            entry["type_counts"][ent_type] += 1
            desc = (ent.get("description") or "").strip()
            if desc:
                entry["descriptions"].append(desc)
            entry["source_chunk_ids"].add(chunk_id)

    result: dict[str, dict[str, Any]] = {}
    for norm_name, entry in groups.items():
        majority_type = entry["type_counts"].most_common(1)[0][0]
        unique_descs = list(dict.fromkeys(entry["descriptions"]))
        description = " | ".join(unique_descs) if unique_descs else ""

        result[norm_name] = {
            "entity_id": entry["entity_id"],
            "name": norm_name,
            "type": majority_type,
            "description": description,
            "source_chunk_ids": sorted(entry["source_chunk_ids"]),
        }
        logger.debug(
            "Entity %s: type=%s (from %s), %d chunks, %d descs merged",
            norm_name, majority_type, dict(entry["type_counts"]),
            len(entry["source_chunk_ids"]), len(unique_descs),
        )

    return result


def dedup_relations(
    raw_extractions: list[dict[str, Any]],
    entity_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedup relations across raw extractions, mapping names to entity_ids.

    Two relations are duplicates when they share the same
    (normalized_source, normalized_target, relation_type). Duplicates are
    merged by concatenating descriptions and unioning source_chunk_ids.

    Args:
        raw_extractions: List of per-chunk {chunk_id, entities, relationships}.
        entity_map: Merged entity registry from ``merge_entities``.

    Returns:
        List of deduped relation dicts with {relation_id, src_entity_id,
        dst_entity_id, type, description, source_chunk_ids}.
    """
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}

    for raw in raw_extractions:
        chunk_id = raw["chunk_id"]
        for rel in raw.get("relationships", []):
            src_raw = rel.get("source", "")
            dst_raw = rel.get("target", "")
            keywords = rel.get("keywords", "")

            src_norm = normalize_entity_name(src_raw)
            dst_norm = normalize_entity_name(dst_raw)
            if not src_norm or not dst_norm:
                continue

            if src_norm not in entity_map:
                logger.warning(
                    "Relation source entity %r not in entity registry "
                    "(chunk %s, relation %s → %s). Skipping.",
                    src_norm, chunk_id, src_raw, dst_raw,
                )
                continue
            if dst_norm not in entity_map:
                logger.warning(
                    "Relation target entity %r not in entity registry "
                    "(chunk %s, relation %s → %s). Skipping.",
                    dst_norm, chunk_id, src_raw, dst_raw,
                )
                continue

            rel_type = keywords.strip()
            if not rel_type:
                rel_type = "related_to"

            key = (src_norm, dst_norm, rel_type)

            if key not in groups:
                groups[key] = {
                    "src_entity_name": src_norm,
                    "dst_entity_name": dst_norm,
                    "type": rel_type,
                    "descriptions": [],
                    "source_chunk_ids": set(),
                }

            desc = (rel.get("description") or "").strip()
            if desc:
                groups[key]["descriptions"].append(desc)
            groups[key]["source_chunk_ids"].add(chunk_id)

    result: list[dict[str, Any]] = []
    for (src_norm, dst_norm, rel_type), entry in groups.items():
        src_id = entity_map[src_norm]["entity_id"]
        dst_id = entity_map[dst_norm]["entity_id"]
        unique_descs = list(dict.fromkeys(entry["descriptions"]))
        description = " | ".join(unique_descs) if unique_descs else ""

        result.append(
            {
                "relation_id": _relation_id(src_id, dst_id, rel_type),
                "src_entity_id": src_id,
                "dst_entity_id": dst_id,
                "type": rel_type,
                "description": description,
                "source_chunk_ids": sorted(entry["source_chunk_ids"]),
            }
        )

    return result
