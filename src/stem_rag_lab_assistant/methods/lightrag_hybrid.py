"""LightRAG-Hybrid — RRF seeds + one-hop expansion over LightRAG's persisted graph.

Reads LightRAG's graph (``lightrag_data/``) without importing ``lightrag-hku``.
Conservative-A relation expansion recovers chunks truncated by the neighbour cap.
Budget = 6 seeds + 4 reranked = 10 chunks (parity with LightRAG chunk_top_k=10).

Note: ``lightrag_data/`` is read-only for this method.  The baseline adapter
(``lightrag_baseline.py``) writes to it; do not re-run baseline insert while
this method's results are being collected.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import QueryFusionRetriever

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
)
from stem_rag_lab_assistant.index.bm25_store import get_bm25_retriever
from stem_rag_lab_assistant.index.graph_store import load_chunk_texts
from stem_rag_lab_assistant.index.vector_store import VectorRetriever
from stem_rag_lab_assistant.methods.common import (
    budget_graph_text,
    format_retrieval_context,
)
from stem_rag_lab_assistant.resources import (
    get_embed_model,
    get_fusion_retriever,
    get_lightrag_graph_store,
    get_reranker,
    get_vector_store,
)

logger = logging.getLogger(__name__)

_CFG = get_config()


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ch in chunks:
        parts.append(f"[{ch['chunk_id']}]\n{ch['text']}")
    return "\n\n".join(parts)


def lightrag_hybrid_retrieve(
    query: str,
) -> dict[str, Any]:
    """Seeds = RRF(vector U BM25) -> map to LightRAG entities -> one-hop expansion
    -> Conservative-A relation expansion -> dedup + cap -> CE rerank.

    Seed budget is identical to hybrid method (RRF top-k, same k).
    Graph expansion is capped at ``RAGConfig.graph_max_expanded_chunks``.

    Returns dict with keys:
        seed_chunks, expanded_chunk_ids, relation_expanded_chunk_ids,
        all_chunks, entities_matched
    """
    t0 = time.perf_counter()
    retriever = get_fusion_retriever()
    graph_store = get_lightrag_graph_store()

    # --- 1. Seeds via RRF (same as hybrid method) ---
    nodes_with_scores = retriever.retrieve(query)
    seed_chunks: list[dict[str, Any]] = []
    seed_ids: list[str] = []
    for nws in nodes_with_scores:
        seed_chunks.append(
            {
                "chunk_id": nws.node.node_id,
                "text": nws.node.text or "",
                "score": float(nws.score) if nws.score is not None else None,
                "metadata": dict(nws.node.metadata or {}),
            }
        )
        seed_ids.append(nws.node.node_id)

    logger.info("Seeds (RRF): %d chunks", len(seed_chunks))

    # --- 2. Map seed chunks -> entities ---
    seed_entities = graph_store.get_entities_for_chunks(seed_ids)
    seed_entity_names = {e["name"] for e in seed_entities}
    logger.info("Seed entities matched: %d", len(seed_entities))

    # --- 3a. Entity-neighbour expansion, capped ---
    entity_neighbour_chunks = graph_store.expand_one_hop(seed_entity_names)
    logger.debug(
        "Entity-neighbour expansion: %d chunk_ids", len(entity_neighbour_chunks),
    )

    # --- 3b. Relation-neighbour expansion (Conservative A) ---
    relation_neighbour_chunks = graph_store.expand_relations(seed_entity_names)
    logger.debug(
        "Relation-neighbour expansion: %d chunk_ids", len(relation_neighbour_chunks),
    )

    # --- 4. Union + dedup against seeds + cap ---
    expanded_pool = entity_neighbour_chunks | relation_neighbour_chunks
    new_ids = sorted(expanded_pool - set(seed_ids))
    relation_expanded_ids = sorted(relation_neighbour_chunks - set(seed_ids))

    new_ids = new_ids[: _CFG.graph_max_expanded_chunks]

    logger.info(
        "One-hop expansion: %d entity + %d relation -> %d union -> "
        "%d after seed dedup (capped at %d)",
        len(entity_neighbour_chunks),
        len(relation_neighbour_chunks),
        len(expanded_pool),
        len(new_ids),
        _CFG.graph_max_expanded_chunks,
    )

    # --- 5. Load neighbour chunk texts ---
    neighbour_chunk_ids = set(new_ids)
    neighbour_texts = load_chunk_texts(neighbour_chunk_ids) if neighbour_chunk_ids else {}
    expanded_chunks: list[dict[str, Any]] = []
    for cid in new_ids:
        text = neighbour_texts.get(cid, "")
        if text:
            expanded_chunks.append(
                {"chunk_id": cid, "text": text, "score": None, "metadata": {}}
            )

    # --- 6. Cross-encoder only on expanded chunks -> seeds always preserved ---
    reranker = get_reranker()
    kept_expanded = reranker.rerank(query, expanded_chunks, top_n=_CFG.rerank_top_n) if expanded_chunks else []
    all_chunks: list[dict[str, Any]] = list(seed_chunks) + kept_expanded

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Retrieval: %d seeds + %d expanded -> CE kept %d/%d = %d total (%.0fms)",
        len(seed_chunks), len(expanded_chunks), len(kept_expanded),
        len(expanded_chunks), len(all_chunks), elapsed,
    )

    return {
        "seed_chunks": seed_chunks,
        "expanded_chunk_ids": [ch["chunk_id"] for ch in kept_expanded],
        "relation_expanded_chunk_ids": [
            cid for cid in relation_expanded_ids
            if cid in {ch["chunk_id"] for ch in kept_expanded}
        ],
        "all_chunks": all_chunks,
        "entities_matched": len(seed_entities),
    }


def lightrag_hybrid_answer(query: str) -> dict[str, Any]:
    """Full LightRAG-hybrid pipeline: retrieve + expand + synthesise answer.

    Returns dict with keys: query, retrieved_chunks, graph_expanded_chunk_ids,
    relation_expanded_chunk_ids, entities_matched, answer, latency_ms, model.
    """
    t0 = time.perf_counter()

    retrieval = lightrag_hybrid_retrieve(query)
    all_chunks = retrieval["all_chunks"]

    context_str = _format_context(all_chunks)
    llm = get_answer_llm()

    messages = [
        ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=ANSWER_USER_TEMPLATE.format(
                context=context_str, query=query,
            ),
        ),
    ]

    response = llm.chat(messages)
    answer = (
        response.message.content
        if hasattr(response, "message")
        else str(response)
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    result = {
        "query": query,
        "retrieved_chunks": [
            {
                "chunk_id": ch["chunk_id"],
                "text": ch["text"],
                "score": ch.get("rerank_score", ch.get("score")),
            }
            for ch in all_chunks
        ],
        "graph_expanded_chunk_ids": retrieval["expanded_chunk_ids"],
        "relation_expanded_chunk_ids": retrieval["relation_expanded_chunk_ids"],
        "entities_matched": retrieval["entities_matched"],
        "seed_chunks_pre_rerank": len(retrieval["seed_chunks"]),
        "context": context_str,
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
        "model": "lightrag_hybrid",
    }
    return result


# ---------------------------------------------------------------------------
# Phase E — LightRAG-Hybrid v2
# Lexical-bridged graph retrieval + full-pool cross-encoder rerank. The graph
# is reached via retrieved seed chunks -> entity *lookup* (no query->entity
# dense search, no per-query keyword LLM). Seeds can be dropped by the CE
# (unlike v1, which always kept them).
# ---------------------------------------------------------------------------

_LH_V2_SEED_RETRIEVER: QueryFusionRetriever | None = None


def _get_lh_v2_seed_retriever() -> QueryFusionRetriever:
    """RRF(vector ∪ BM25) seeder for v2 — fusion top-k = ``lh_seed_k``.

    Owns its fusion retriever so widening the fused seed set does not leak into
    the shared singleton used by the other methods.
    """
    global _LH_V2_SEED_RETRIEVER
    if _LH_V2_SEED_RETRIEVER is not None:
        return _LH_V2_SEED_RETRIEVER
    vector_retriever = VectorRetriever(
        get_vector_store(), get_embed_model(), top_k=_CFG.vector_top_k,
    )
    _LH_V2_SEED_RETRIEVER = QueryFusionRetriever(
        [vector_retriever, get_bm25_retriever()],
        similarity_top_k=_CFG.lh_seed_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=get_answer_llm(),
    )
    logger.info("LH-v2 seed retriever ready (RRF fusion top_k=%d)", _CFG.lh_seed_k)
    return _LH_V2_SEED_RETRIEVER


def _cosine_pool_rank(query: str, chunk_ids: list[str], cap: int) -> list[str]:
    """Rank pool chunk ids by cosine(query, chunk embedding); keep top *cap*.

    Cheap pre-CE bounding (in-memory embeddings, no disk I/O) so the
    cross-encoder only scores plausible candidates.
    """
    q = np.asarray(get_embed_model().get_query_embedding(query), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    emb = get_vector_store()._data.embedding_dict

    scored: list[tuple[str, float]] = []
    for cid in chunk_ids:
        vec = emb.get(cid)
        if vec is None:
            scored.append((cid, -1.0))
            continue
        v = np.asarray(vec, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        scored.append((cid, float(v @ q)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:cap]]


def lightrag_hybrid_v2_retrieve(query: str) -> dict[str, Any]:
    """RRF seeds -> seed-chunk→entity lookup -> one-hop expansion -> cosine
    pool cap -> full-pool CE rerank -> top-N + matched (graph-text) context.
    """
    t0 = time.perf_counter()
    graph_store = get_lightrag_graph_store()

    # 1. Lexical/vector RRF seeds (the only retrieval-side "search").
    nodes = _get_lh_v2_seed_retriever().retrieve(query)
    seed_chunks = [
        {
            "chunk_id": n.node.node_id,
            "text": n.node.text or "",
            "score": float(n.score) if n.score is not None else None,
            "metadata": dict(n.node.metadata or {}),
        }
        for n in nodes
    ]
    seed_ids = [c["chunk_id"] for c in seed_chunks]

    # 2. Seed chunks -> entities (lookup via stored chunk↔entity map), capped.
    seed_entities = graph_store.get_entities_for_chunks(seed_ids)
    seed_entity_names = [e["name"] for e in seed_entities][: _CFG.lh_seed_entity_cap]
    nameset = set(seed_entity_names)

    # 3. One-hop entity + relation expansion.
    entity_neigh = graph_store.expand_one_hop(
        nameset,
        neighbour_cap=_CFG.lh_neighbour_cap,
        entity_cap=_CFG.lh_seed_entity_cap,
        chunk_cap_per_neighbour=_CFG.lh_neighbour_cap,
    )
    relation_neigh = graph_store.expand_relations(nameset)

    # 4. Union + cosine pre-rank + pool cap.
    pool = {c for c in (set(seed_ids) | entity_neigh | relation_neigh) if c}
    if len(pool) > _CFG.lh_pool_cap:
        pool = set(_cosine_pool_rank(query, sorted(pool), _CFG.lh_pool_cap))

    texts = load_chunk_texts(pool)
    pool_chunks = [
        {"chunk_id": cid, "text": texts[cid]} for cid in pool if cid in texts
    ]

    # 5. Full-pool cross-encoder rerank -> top-N (seeds may be dropped).
    reranker = get_reranker()
    ranked = reranker.rerank(query, pool_chunks, top_n=_CFG.lh_topn)
    ranked_ids = {c["chunk_id"] for c in ranked}

    # 6. Matched context: graph text describes the evidence actually returned —
    # entities behind the final top-N chunks (ordered by how many chunks they
    # support) plus relations among those entities. Lookup/adjacency only.
    graph_text = ""
    if _CFG.lh_include_graph_text:
        counts = graph_store.get_entity_names_for_chunks(
            [c["chunk_id"] for c in ranked]
        )
        context_names = sorted(counts, key=lambda n: (-counts[n], n))
        ents = graph_store.get_entity_descriptions(context_names)
        rels = graph_store.get_relations_among(context_names)
        graph_text = budget_graph_text(ents, rels)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "LH-v2 retrieval: %d seeds + %d expanded -> pool %d -> CE kept %d "
        "(%.0fms, graph_text=%d chars)",
        len(seed_chunks),
        len(pool) - len(set(seed_ids) & pool),
        len(pool_chunks),
        len(ranked),
        elapsed,
        len(graph_text),
    )

    return {
        "all_chunks": ranked,
        "graph_text": graph_text,
        "seed_chunks": seed_chunks,
        "expanded_chunk_ids": [cid for cid in ranked_ids if cid not in set(seed_ids)],
        "relation_expanded_chunk_ids": [cid for cid in ranked_ids if cid in relation_neigh],
        "entities_matched": len(seed_entities),
    }


def lightrag_hybrid_v2_answer(query: str) -> dict[str, Any]:
    """Full v2 pipeline: retrieve + matched context + shared Groq synthesis."""
    t0 = time.perf_counter()
    retrieval = lightrag_hybrid_v2_retrieve(query)
    chunks = retrieval["all_chunks"]
    context_str = format_retrieval_context(chunks, retrieval["graph_text"])

    llm = get_answer_llm()
    messages = [
        ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=ANSWER_USER_TEMPLATE.format(context=context_str, query=query),
        ),
    ]
    response = llm.chat(messages)
    answer = (
        response.message.content if hasattr(response, "message") else str(response)
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": query,
        "retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "score": c.get("rerank_score", c.get("score")),
            }
            for c in chunks
        ],
        "graph_expanded_chunk_ids": retrieval["expanded_chunk_ids"],
        "relation_expanded_chunk_ids": retrieval["relation_expanded_chunk_ids"],
        "entities_matched": retrieval["entities_matched"],
        "context": context_str,
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
        "model": "lightrag_hybrid_v2",
        # Efficiency: the lexical bridge makes no retrieval-time LLM calls.
        "retrieval_llm_calls": 0,
        "retrieval_prompt_tokens": 0,
        "retrieval_completion_tokens": 0,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sample_questions = [
        "What is the difference between accuracy and precision in measurements?",
        "Explain how a Wheatstone bridge works for measuring unknown resistance.",
        "How does an ADC converter relate to quantization error?",
    ]

    for q in sample_questions:
        result = lightrag_hybrid_answer(q)
        print(f"\n{'=' * 80}")
        print(f"Q: {result['query']}")
        print(
            f"Seeds: {len(result['retrieved_chunks'])} "
            + f"+ graph expanded: {len(result['graph_expanded_chunk_ids'])} "
            + f"+ relation expanded: {len(result['relation_expanded_chunk_ids'])} "
            + f"(matched {result['entities_matched']} entities) "
            + f"(latency={result['latency_ms']}ms)"
        )
        for ch in result["retrieved_chunks"]:
            print(f"  {ch['chunk_id']}: {ch['text'][:100]}...")
        for cid in result["graph_expanded_chunk_ids"]:
            print(f"  graph: {cid}")
        for cid in result["relation_expanded_chunk_ids"]:
            print(f"  rel: {cid}")
        print(f"\nA: {result['answer']}")
