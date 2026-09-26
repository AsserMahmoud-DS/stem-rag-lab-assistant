"""Graph store — load graph.json, on-the-fly chunk→entity mapping, one-hop expansion. (P6)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import (
    GRAPH_EXPANSION_DEPTH,
    GRAPH_NEIGHBOUR_CAP,
    GRAPH_SEED_ENTITIES_CAP,
)
from stem_rag_lab_assistant.graph.build import GRAPH_JSON_PATH

logger = logging.getLogger(__name__)


class GraphStore:
    """Query-side graph reader — loads graph.json and answers neighbour queries.

    Per the plan (§14.3), no stored ``chunk_to_entities`` inverted index:
    seed→entity mapping is computed on-the-fly from
    ``entities[].source_chunk_ids`` (corpus is small, scan is microseconds).
    """

    def __init__(self, graph: dict[str, Any]) -> None:
        self._graph = graph
        self._entity_index: dict[str, dict[str, Any]] = {
            e["entity_id"]: e for e in graph.get("entities", [])
        }
        # undirected adjacency: entity_id → set of neighbour entity_ids
        self._adjacency: dict[str, set[str]] = {}
        for r in graph.get("relations", []):
            src = r["src_entity_id"]
            dst = r["dst_entity_id"]
            self._adjacency.setdefault(src, set()).add(dst)
            self._adjacency.setdefault(dst, set()).add(src)

        logger.info(
            "GraphStore loaded: %d entities, %d relations",
            len(self._entity_index),
            len(graph.get("relations", [])),
        )

    @property
    def entities(self) -> list[dict[str, Any]]:
        return self._graph["entities"]

    @property
    def relations(self) -> list[dict[str, Any]]:
        return self._graph["relations"]

    def get_entities_for_chunks(
        self, chunk_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Find entities whose ``source_chunk_ids`` intersect *chunk_ids*.

        Computed on-the-fly — no stored inverted index.
        """
        cid_set = set(chunk_ids)
        matched: list[dict[str, Any]] = []
        for entity in self._graph.get("entities", []):
            if cid_set.intersection(entity.get("source_chunk_ids", [])):
                matched.append(entity)
        logger.debug(
            "%d entities matched from %d seed chunk_ids",
            len(matched), len(chunk_ids),
        )
        return matched

    def expand_one_hop(
        self,
        entity_ids: list[str],
        *,
        neighbour_cap: int = GRAPH_NEIGHBOUR_CAP,
        entity_cap: int = GRAPH_SEED_ENTITIES_CAP,
        chunk_cap_per_neighbour: int = GRAPH_NEIGHBOUR_CAP,
    ) -> set[str]:
        """One-hop expansion: for each seed entity, collect neighbour
        entity chunks via relation edges.

        Returns a set of **chunk_ids** from neighbour entities (NOT the
        seed entities' own chunks).
        """
        neighbour_chunks: set[str] = set()
        for eid in entity_ids[:entity_cap]:
            neighbours = self._adjacency.get(eid, set())
            for nid in list(neighbours)[:neighbour_cap]:
                neighbour = self._entity_index.get(nid)
                if neighbour is None:
                    continue
                for cid in neighbour.get("source_chunk_ids", [])[
                    :chunk_cap_per_neighbour
                ]:
                    neighbour_chunks.add(cid)

        logger.debug(
            "One-hop expansion: %d seed entities → %d neighbour chunk_ids",
            min(len(entity_ids), entity_cap), len(neighbour_chunks),
        )
        return neighbour_chunks

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        return self._entity_index.get(entity_id)


def load_graph_store(path: Path | None = None) -> GraphStore:
    """Load graph.json and return a GraphStore instance."""
    p = path or GRAPH_JSON_PATH
    with open(p, "r", encoding="utf-8") as f:
        graph = json.load(f)
    return GraphStore(graph)


def load_chunk_texts(
    chunk_ids: set[str],
    chunks_json_path: Path | None = None,
) -> dict[str, str]:
    """Look up chunk texts from chunks.json by chunk_id."""
    if chunks_json_path is None:
        chunks_json_path = Path(__file__).resolve().parents[3] / "chunks.json"
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: dict[str, str] = {}
    for doc in data.get("docs", {}).values():
        for ch in doc.get("chunks", []):
            if ch["chunk_id"] in chunk_ids:
                texts[ch["chunk_id"]] = ch.get("text", "")
    return texts
