"""Hybrid+Graph RAG — seeds via RRF → one-hop graph expansion (our contribution, NO lightrag-hku import). (P6)"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
)
from stem_rag_lab_assistant.index.bm25_store import get_bm25_retriever
from stem_rag_lab_assistant.index.graph_store import (
    GraphStore,
    load_chunk_texts,
    load_graph_store,
)
from stem_rag_lab_assistant.index.vector_store import (
    VectorRetriever,
    load_vector_store,
)
from stem_rag_lab_assistant.index.reranker import CrossEncoderReranker, get_reranker

logger = logging.getLogger(__name__)

_CFG = get_config()

_EMBED_MODEL: HuggingFaceEmbedding | None = None
_FUSION_RETRIEVER: QueryFusionRetriever | None = None
_GRAPH_STORE: GraphStore | None = None


def _get_embed_model() -> HuggingFaceEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=_CFG.embedding_model, device="cuda",
        )
    return _EMBED_MODEL


def _get_fusion_retriever() -> QueryFusionRetriever:
    global _FUSION_RETRIEVER
    if _FUSION_RETRIEVER is not None:
        return _FUSION_RETRIEVER

    embed_model = _get_embed_model()
    vector_store = load_vector_store()
    vector_retriever = VectorRetriever(vector_store, embed_model, top_k=_CFG.vector_top_k)
    bm25_retriever = get_bm25_retriever()

    _FUSION_RETRIEVER = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=_CFG.vector_top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=get_answer_llm(),
    )
    logger.info("Graph seed fusion retriever ready (RRF, top_k=%d)", _CFG.vector_top_k)
    return _FUSION_RETRIEVER


def _get_graph_store() -> GraphStore:
    global _GRAPH_STORE
    if _GRAPH_STORE is None:
        _GRAPH_STORE = load_graph_store()
    return _GRAPH_STORE


_reranker: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = get_reranker()
    return _reranker


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ch in chunks:
        parts.append(f"[{ch['chunk_id']}]\n{ch['text']}")
    return "\n\n".join(parts)


def hybrid_graph_retrieve(
    query: str,
) -> dict[str, Any]:
    """Seeds = RRF(vector ∪ BM25) → map to entities → one-hop expansion.

    Seed budget is identical to the hybrid method (RRF top-k, same k).
    Graph expansion is capped at ``RAGConfig.graph_max_expanded_chunks`` to
    control context cost while still demonstrating cross-document reach.

    Returns dict with keys:
        seed_chunks, expanded_chunk_ids, all_chunks, entities_matched
    """
    t0 = time.perf_counter()
    retriever = _get_fusion_retriever()
    graph_store = _get_graph_store()

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

    # --- 2. Map seed chunks → entities ---
    seed_entities = graph_store.get_entities_for_chunks(seed_ids)
    seed_entity_ids = [e["entity_id"] for e in seed_entities]
    logger.info("Seed entities matched: %d", len(seed_entities))

    # --- 3. One-hop expansion, capped ---
    expanded_chunk_ids = graph_store.expand_one_hop(seed_entity_ids)
    new_ids = sorted(expanded_chunk_ids - set(seed_ids))

    new_ids = new_ids[: _CFG.graph_max_expanded_chunks]

    logger.info(
        "One-hop expansion: %d neighbour chunks (%d added, capped at %d)",
        len(expanded_chunk_ids), len(new_ids), _CFG.graph_max_expanded_chunks,
    )

    # --- 4. Load neighbour chunk texts (separate from seeds) ---
    neighbour_texts = load_chunk_texts(set(new_ids)) if new_ids else {}
    expanded_chunks: list[dict[str, Any]] = []
    for cid in new_ids:
        text = neighbour_texts.get(cid, "")
        if text:
            expanded_chunks.append(
                {"chunk_id": cid, "text": text, "score": None, "metadata": {}}
            )

    # --- 5. Cross-encoder only on expanded chunks → seeds always preserved ---
    reranker = _get_reranker()
    kept_expanded = reranker.rerank(query, expanded_chunks, top_n=_CFG.rerank_top_n) if expanded_chunks else []
    all_chunks: list[dict[str, Any]] = list(seed_chunks) + kept_expanded

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Retrieval: %d seeds + %d expanded → CE kept %d/%d = %d total (%.0fms)",
        len(seed_chunks), len(expanded_chunks), len(kept_expanded),
        len(expanded_chunks), len(all_chunks), elapsed,
    )

    return {
        "seed_chunks": seed_chunks,
        "expanded_chunk_ids": [ch["chunk_id"] for ch in kept_expanded],
        "all_chunks": all_chunks,
        "entities_matched": len(seed_entities),
    }


def hybrid_graph_answer(query: str) -> dict[str, Any]:
    """Full hybrid+graph RAG pipeline: retrieve + expand + synthesise answer.

    Returns dict with keys: query, retrieved_chunks, graph_expanded_chunk_ids,
    entities_matched, answer, latency_ms, model.
    """
    t0 = time.perf_counter()

    retrieval = hybrid_graph_retrieve(query)
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
            {"chunk_id": ch["chunk_id"], "text": ch["text"], "score": ch.get("rerank_score", ch.get("score"))}
            for ch in all_chunks
        ],
        "graph_expanded_chunk_ids": retrieval["expanded_chunk_ids"],
        "entities_matched": retrieval["entities_matched"],
        "seed_chunks_pre_rerank": len(retrieval["seed_chunks"]),
        "context": context_str,
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
        "model": "hybrid_graph",
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
        result = hybrid_graph_answer(q)
        print(f"\n{'=' * 80}")
        print(f"Q: {result['query']}")
        print(
            f"Seeds: {len(result['retrieved_chunks'])} "
            f"+ graph expanded: {len(result['graph_expanded_chunk_ids'])} "
            f"(matched {result['entities_matched']} entities) "
            f"(latency={result['latency_ms']}ms)"
        )
        for ch in result["retrieved_chunks"]:
            print(f"  seed: {ch['chunk_id']}: {ch['text'][:100]}...")
        for cid in result["graph_expanded_chunk_ids"]:
            print(f"  graph: {cid}")
        print(f"\nA: {result['answer']}")
