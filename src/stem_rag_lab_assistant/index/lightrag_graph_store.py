"""LightRAG graph store — reads persisted LightRAG artifacts (NO lightrag-hku import).

Reads graphml + kv_store JSONs from ``lightrag_data/`` directly via ``networkx``
and ``json``.  Implements the same interface as ``index/graph_store.py:GraphStore``
so that ``lightrag_hybrid.py`` is policy-identical to ``hybrid_graph.py`` except
for graph source and the Conservative-A relation expansion.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import networkx

from stem_rag_lab_assistant.config import (
    GRAPH_NEIGHBOUR_CAP,
    GRAPH_SEED_ENTITIES_CAP,
    LIGHTRAG_WORKING_DIR,
)

logger = logging.getLogger(__name__)

_CHUNK_SUFFIX_RE = re.compile(r"-chunk-\d+$")


def _normalize_chunk_id(cid: str) -> str:
    """Strip LightRAG's ``-chunk-NNN`` suffix so chunk_ids match chunks.json."""
    return _CHUNK_SUFFIX_RE.sub("", cid)


class LightRAGGraphStore:
    """Query-side graph reader over LightRAG's persisted artifacts.

    Reads ``graph_chunk_entity_relation.graphml`` (entity adjacency) plus
    ``kv_store_entity_chunks.json`` (inverted → chunk→entities) and
    ``kv_store_relation_chunks.json`` (Conservative-A relation expansion)
    from ``lightrag_data/``.

    Same interface as ``index/graph_store.py:GraphStore``.
    """

    def __init__(self, working_dir: Path) -> None:
        self._working_dir = working_dir
        self._entity_nodes: dict[str, dict[str, Any]] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._chunk_to_entities: dict[str, set[str]] = {}
        self._relation_keys: dict[str, set[str]] = {}
        self._relation_chunks: dict[str, set[str]] = {}

        self._load_graphml()
        self._load_entity_chunks()
        self._load_relation_chunks()

        logger.info(
            "LightRAGGraphStore loaded: %d entities, %d adjacency entries, "
            "%d inverted chunk→entity entries, %d relations",
            len(self._entity_nodes),
            sum(len(v) for v in self._adjacency.values()),
            len(self._chunk_to_entities),
            len(self._relation_chunks),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_graphml(self) -> None:
        graphml_path = self._working_dir / "graph_chunk_entity_relation.graphml"
        if not graphml_path.exists():
            raise FileNotFoundError(
                f"LightRAG graphml not found at {graphml_path}. "
                "Run the lightrag baseline insert first."
            )

        g = networkx.read_graphml(graphml_path)

        for node_id, data in g.nodes(data=True):
            entity_id = data.get("entity_id", node_id)
            self._entity_nodes[entity_id] = dict(data)
            self._adjacency.setdefault(entity_id, set())

        for src, dst, data in g.edges(data=True):
            self._adjacency.setdefault(src, set()).add(dst)
            self._adjacency.setdefault(dst, set()).add(src)

        logger.debug("graphml: %d nodes, %d edges loaded", len(g.nodes), len(g.edges))

    def _load_entity_chunks(self) -> None:
        path = self._working_dir / "kv_store_entity_chunks.json"
        if not path.exists():
            logger.warning("kv_store_entity_chunks.json not found at %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entity_name, entry in data.items():
            for raw_cid in entry.get("chunk_ids", []):
                cid = _normalize_chunk_id(raw_cid)
                self._chunk_to_entities.setdefault(cid, set()).add(entity_name)

        logger.debug("entity_chunks: %d chunk→entity entries", len(self._chunk_to_entities))

    def _load_relation_chunks(self) -> None:
        path = self._working_dir / "kv_store_relation_chunks.json"
        if not path.exists():
            logger.warning("kv_store_relation_chunks.json not found at %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for relation_key, entry in data.items():
            chunks = {_normalize_chunk_id(cid) for cid in entry.get("chunk_ids", [])}
            if not chunks:
                continue

            self._relation_chunks[relation_key] = chunks

            if "<SEP>" in relation_key:
                src, dst = relation_key.split("<SEP>", 1)
                self._relation_keys.setdefault(src, set()).add(relation_key)
                self._relation_keys.setdefault(dst, set()).add(relation_key)

        logger.debug(
            "relation_chunks: %d relations, %d entity→relation entries",
            len(self._relation_chunks),
            sum(len(v) for v in self._relation_keys.values()),
        )

    # ------------------------------------------------------------------
    # Query interface  (compatible with index/graph_store.py:GraphStore)
    # ------------------------------------------------------------------

    def get_entities_for_chunks(
        self, chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Find entities whose source_chunk_ids intersect *chunk_ids*.

        Returns list of dicts with ``entity_id``, ``name``, ``type``,
        ``description``, ``source_chunk_ids`` — same keys as
        ``GraphStore.get_entities_for_chunks``.
        """
        cid_set = set(chunk_ids)
        matched: list[dict[str, Any]] = []

        for entity_name in sorted(set().union(
            *(self._chunk_to_entities.get(cid, set()) for cid in cid_set),
        )):
            node = self._entity_nodes.get(entity_name, {})
            source_ids_raw = node.get("source_id", "")
            source_chunk_ids = sorted({
                _normalize_chunk_id(cid)
                for cid in source_ids_raw.split("<SEP>")
                if cid.strip()
            })

            matched.append(
                {
                    "entity_id": entity_name,
                    "name": entity_name,
                    "type": node.get("entity_type", ""),
                    "description": node.get("description", ""),
                    "source_chunk_ids": source_chunk_ids,
                }
            )

        logger.debug(
            "%d entities matched from %d seed chunk_ids",
            len(matched), len(chunk_ids),
        )
        return matched

    def expand_one_hop(
        self,
        entity_names: set[str],
        neighbour_cap: int = GRAPH_NEIGHBOUR_CAP,
        entity_cap: int = GRAPH_SEED_ENTITIES_CAP,
        chunk_cap_per_neighbour: int = GRAPH_NEIGHBOUR_CAP,
    ) -> set[str]:
        """One-hop expansion: for each seed entity, collect neighbour
        entity chunks via relation edges.

        Returns a set of **chunk_ids** from neighbour entities.
        """
        neighbour_chunks: set[str] = set()

        for ename in sorted(entity_names)[:entity_cap]:
            neighbours = self._adjacency.get(ename, set())
            for nid in sorted(neighbours)[:neighbour_cap]:
                node = self._entity_nodes.get(nid)
                if node is None:
                    continue
                source_ids_raw = node.get("source_id", "")
                chunks = {
                    _normalize_chunk_id(cid)
                    for cid in source_ids_raw.split("<SEP>")
                    if cid.strip()
                }
                for cid in sorted(chunks)[:chunk_cap_per_neighbour]:
                    neighbour_chunks.add(cid)

        logger.debug(
            "One-hop expansion: %d seed entities → %d neighbour chunk_ids",
            min(len(entity_names), entity_cap),
            len(neighbour_chunks),
        )
        return neighbour_chunks

    def expand_relations(
        self, seed_entity_names: set[str],
    ) -> set[str]:
        """Conservative-A relation expansion.

        For each relation where a seed entity appears as src or dst,
        return the union of that relation's chunk_ids.

        This recovers chunks that the neighbour cap truncated in
        ``expand_one_hop`` — relations touch the seed entity directly
        and are not subject to the neighbour cap.
        """
        relation_chunks: set[str] = set()

        for ename in seed_entity_names:
            for rkey in self._relation_keys.get(ename, set()):
                relation_chunks |= self._relation_chunks.get(rkey, set())

        logger.debug(
            "Relation expansion: %d seed entities → %d relation chunk_ids",
            len(seed_entity_names),
            len(relation_chunks),
        )
        return relation_chunks

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Return entity node data by name."""
        return self._entity_nodes.get(entity_id)


# ------------------------------------------------------------------
# Singleton loader
# ------------------------------------------------------------------

_lightrag_graph_store: LightRAGGraphStore | None = None


def load_lightrag_graph_store(
    working_dir: Path | None = None,
) -> LightRAGGraphStore:
    """Load LightRAGGraphStore singleton from persisted artifacts."""
    global _lightrag_graph_store
    if _lightrag_graph_store is None:
        wd = working_dir or LIGHTRAG_WORKING_DIR
        _lightrag_graph_store = LightRAGGraphStore(wd)
    return _lightrag_graph_store


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    store = load_lightrag_graph_store()

    # Test entity lookup
    entities = store.get_entities_for_chunks(["chap 3 DC&AC BRIDGES::ch_00042"])
    print(f"\nEntities for 'chap 3 DC&AC BRIDGES::ch_00042': {len(entities)}")
    for e in entities[:5]:
        print(f"  {e['name']} ({e['type']}): {e['description'][:100]}...")

    # Test one-hop expansion
    if "Wheatstone Bridge" in store._entity_nodes:
        one_hop = store.expand_one_hop({"Wheatstone Bridge"})
        print(f"\nOne-hop from 'Wheatstone Bridge': {len(one_hop)} chunk_ids")
        for cid in sorted(one_hop)[:5]:
            print(f"  {cid}")

    # Test relation expansion
    if "Wheatstone Bridge" in store._entity_nodes:
        rel = store.expand_relations({"Wheatstone Bridge"})
        print(f"\nRelation expansion from 'Wheatstone Bridge': {len(rel)} chunk_ids")
        for cid in sorted(rel)[:5]:
            print(f"  {cid}")
