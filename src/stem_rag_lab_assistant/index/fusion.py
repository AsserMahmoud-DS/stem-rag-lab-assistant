"""RRF fusion retriever keyed by chunk identity (``node_id``).

LlamaIndex's ``QueryFusionRetriever`` fuses by ``node.hash``, which is
``sha256(text + metadata)``. Our vector-store nodes and BM25 nodes carry
*slightly different metadata* for the same chunk (the vector store adds
``doc_id``/``document_id``/``ref_doc_id`` placeholders), so one logical chunk
hashes differently across retrievers. Two consequences:

1. RRF never combines a chunk's ranks across retrievers, so agreement between
   vector and lexical retrieval is not rewarded (the fusion is not RRF).
2. The duplicate entries survive the ``[:similarity_top_k]`` slice and are
   only collapsed afterwards by ``BaseRetriever._handle_recursive_retrieval``
   (which dedups on ``node_id``), so the realised candidate pool shrinks below
   the requested ``similarity_top_k``.

This subclass keys fusion on ``node.node_id`` — our ``chunk_id``, the correct
document identity — so cross-retriever agreement boosts the score and the
post-hoc dedup becomes a no-op.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

# RRF's rank constant (the original paper's k=60).
_RRF_K = 60.0


class RRFRetriever(QueryFusionRetriever):
    """Reciprocal-rank fusion over retriever lists, keyed by ``node_id``."""

    def _fusion_by_node_id(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        fused: dict[str, float] = {}
        id_to_node: dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            ranked = sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
            for rank, nws in enumerate(ranked):
                nid = nws.node.node_id
                fused[nid] = fused.get(nid, 0.0) + 1.0 / (rank + _RRF_K)
                if nid not in id_to_node:
                    id_to_node[nid] = nws

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        out: list[NodeWithScore] = []
        for nid, score in ordered:
            nws = id_to_node[nid]
            nws.score = score
            out.append(nws)
        return out

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: list[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        return self._fusion_by_node_id(results)[: self.similarity_top_k]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: list[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(await self._aget_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        return self._fusion_by_node_id(results)[: self.similarity_top_k]
