"""Graph build — assemble graph.json from raw extractions (ONLY writer of graph.json). (P4/P5)"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import ROOT_DIR, get_config
from stem_rag_lab_assistant.graph.merge import dedup_relations, merge_entities

logger = logging.getLogger(__name__)

GRAPH_JSON_PATH = ROOT_DIR / "graph.json"


def _build_corpus_hash(chunks_json_path: Path) -> str:
    """Stable hash from per-doc content_hashes already in chunks.json.

    O(n_docs) instead of O(n_chunks) — reuses the hashes computed by
    ``corpus/ingest.py`` during incremental ingest. Changes if any PDF
    file changes, a doc is added/removed, or the sorted order differs.
    """
    if not chunks_json_path.exists():
        return "sha256:empty"
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = [
        data["docs"][doc_id].get("content_hash", "")
        for doc_id in sorted(data.get("docs", {}).keys())
    ]
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def build_graph(
    raw_extractions: list[dict[str, Any]],
    chunks_json_path: Path | None = None,
    *,
    entity_type_profile: str = "ece-v1",
) -> dict[str, Any]:
    """Build the full graph.json dict from raw per-chunk extractions.

    Args:
        raw_extractions: List of {chunk_id, entities, relationships} per chunk.
        chunks_json_path: Path to chunks.json for the ``built_from_chunks_hash``.
        entity_type_profile: Label for the entity-type prompt profile used.

    Returns:
        graph.json dict per the project schema.
    """
    entity_map = merge_entities(raw_extractions)
    relations = dedup_relations(raw_extractions, entity_map)

    entities_list = sorted(entity_map.values(), key=lambda e: e["entity_id"])
    for ent in entities_list:
        ent.pop("type_counts", None)

    relations_list = sorted(relations, key=lambda r: r["relation_id"])

    corpus_hash = _build_corpus_hash(
        chunks_json_path or ROOT_DIR / "chunks.json"
    )

    graph = {
        "version": 1,
        "extraction_model": get_config().extraction_llm_model,
        "entity_type_profile": entity_type_profile,
        "built_from_chunks_hash": corpus_hash,
        "entities": entities_list,
        "relations": relations_list,
    }

    logger.info(
        "Graph built: %d entities, %d relations (corpus hash: %s)",
        len(entities_list), len(relations_list), corpus_hash,
    )

    return graph


def save_graph(graph: dict[str, Any], path: Path | None = None) -> Path:
    """Persist graph dict to disk as JSON.

    Args:
        graph: Graph dict from ``build_graph``.
        path: Output path (default: ``ROOT_DIR/graph.json``).

    Returns:
        The path written to.
    """
    output_path = path or GRAPH_JSON_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    logger.info("Graph saved to %s", output_path)
    return output_path


def load_graph(path: Path | None = None) -> dict[str, Any]:
    """Load graph.json from disk.

    Args:
        path: Input path (default: ``ROOT_DIR/graph.json``).

    Returns:
        The graph dict.
    """
    input_path = path or GRAPH_JSON_PATH
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
