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

from llama_index.core.llms import ChatMessage

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
)
from stem_rag_lab_assistant.index.graph_store import load_chunk_texts
from stem_rag_lab_assistant.resources import (
    get_fusion_retriever,
    get_lightrag_graph_store,
    get_reranker,
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
