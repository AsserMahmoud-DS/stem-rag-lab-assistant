"""Hybrid RAG — BM25 + vector fusion via id-keyed RRF. (P2)"""

from __future__ import annotations

import logging
import time
from typing import Any

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from stem_rag_lab_assistant.index.bm25_store import get_bm25_retriever_at
from stem_rag_lab_assistant.index.fusion import RRFRetriever
from stem_rag_lab_assistant.index.reranker import get_reranker
from stem_rag_lab_assistant.index.vector_store import VectorRetriever
from stem_rag_lab_assistant.methods.common import answer_usage, format_retrieval_context
from stem_rag_lab_assistant.resources import get_embed_model, get_fusion_retriever, get_vector_store

logger = logging.getLogger(__name__)

_CFG = get_config()


def hybrid_retrieve(query: str) -> list[dict[str, Any]]:
    """Fuse vector + BM25 via RRF; return [{chunk_id, text, score, metadata}]."""
    retriever = get_fusion_retriever()
    nodes_with_scores = retriever.retrieve(query)

    results: list[dict[str, Any]] = []
    for nws in nodes_with_scores:
        results.append({
            "chunk_id": nws.node.node_id,
            "text": nws.node.text or "",
            "score": float(nws.score) if nws.score is not None else None,
            "metadata": dict(nws.node.metadata or {}),
        })

    logger.info("Hybrid retrieval: %d fused results for query: %r", len(results), query[:60])
    for r in results:
        logger.debug("  %s  score=%.4f", r["chunk_id"], r["score"])

    return results


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ch in chunks:
        parts.append(f"[{ch['chunk_id']}]\n{ch['text']}")
    return "\n\n".join(parts)


def hybrid_answer(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full hybrid RAG pipeline: retrieve (RRF) + synthesise answer.

    Returns dict with keys: query, retrieved_chunks, answer, latency_ms, model.
    """
    t0 = time.perf_counter()

    if chunks is None:
        chunks = hybrid_retrieve(query)

    context_str = _format_context(chunks)
    llm = get_answer_llm()

    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=ANSWER_USER_TEMPLATE.format(
            context=context_str, query=query,
        )),
    ]

    response = llm.chat(messages)
    answer = response.message.content if hasattr(response, "message") else str(response)

    latency_ms = (time.perf_counter() - t0) * 1000

    result = {
        "query": query,
        "retrieved_chunks": [
            {"chunk_id": ch["chunk_id"], "text": ch["text"], "score": ch["score"]}
            for ch in chunks
        ],
        "context": context_str,
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
        "model": "hybrid",
    }
    return result


# ---------------------------------------------------------------------------
# Phase E parity control — hybrid_ce
# Same candidate budget (lh_pool_cap), cross-encoder and final budget
# (lh_topn) as lightrag_hybrid_v2, but candidates come from deeper RRF
# retrieval instead of graph one-hop expansion. Chunks-only context.
# ---------------------------------------------------------------------------

_HYBRID_CE_RETRIEVER: RRFRetriever | None = None


def _get_hybrid_ce_retriever() -> RRFRetriever:
    """RRF(vector ∪ BM25) candidate seeder at the shared candidate budget.

    Sub-retrievers are built at ``lh_pool_cap`` too: LlamaIndex's fusion
    queries each sub-retriever with its *own* top-k and only trims the fused
    list, so a k=28 fusion over k=6 sub-retrievers could never reach 28. The
    BM25 instance is private (``get_bm25_retriever_at``) so widening k does
    not leak into the shared singleton used by the legacy hybrid method.
    """
    global _HYBRID_CE_RETRIEVER
    if _HYBRID_CE_RETRIEVER is not None:
        return _HYBRID_CE_RETRIEVER
    vector_retriever = VectorRetriever(
        get_vector_store(), get_embed_model(), top_k=_CFG.lh_pool_cap,
    )
    _HYBRID_CE_RETRIEVER = RRFRetriever(
        [vector_retriever, get_bm25_retriever_at(_CFG.lh_pool_cap)],
        similarity_top_k=_CFG.lh_pool_cap,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=get_answer_llm(),
    )
    logger.info("Hybrid-CE candidate retriever ready (RRF top_k=%d)", _CFG.lh_pool_cap)
    return _HYBRID_CE_RETRIEVER


def hybrid_ce_answer(query: str) -> dict[str, Any]:
    """Budget-parity hybrid control: RRF top-N -> CE -> top-k, chunks-only.

    Identical candidate budget, cross-encoder and final budget as
    ``lightrag_hybrid_v2``; the difference is where candidates come from
    (deeper RRF retrieval vs graph one-hop expansion).
    """
    t0 = time.perf_counter()

    nodes = _get_hybrid_ce_retriever().retrieve(query)
    t_retrieval = time.perf_counter()
    candidates = [
        {
            "chunk_id": n.node.node_id,
            "text": n.node.text or "",
            "score": float(n.score) if n.score is not None else None,
            "metadata": dict(n.node.metadata or {}),
        }
        for n in nodes
    ]

    ranked = get_reranker().rerank(query, candidates, top_n=_CFG.lh_topn)
    t_rerank = time.perf_counter()
    context_str = format_retrieval_context(ranked, "")

    llm = get_answer_llm()
    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=ANSWER_USER_TEMPLATE.format(
            context=context_str, query=query,
        )),
    ]

    response = llm.chat(messages)
    t_answer = time.perf_counter()
    answer = response.message.content if hasattr(response, "message") else str(response)
    prompt_tok, completion_tok, reasoning_tok = answer_usage(response)

    return {
        "query": query,
        "retrieved_chunks": [
            {"chunk_id": ch["chunk_id"], "text": ch["text"], "score": ch["rerank_score"]}
            for ch in ranked
        ],
        "context": context_str,
        "answer": answer,
        "latency_ms": round((t_answer - t0) * 1000, 1),
        "model": "hybrid_ce",
        # Local retrieval (RRF + cross-encoder): no retrieval-side LLM calls.
        "retrieval_llm_calls": 0,
        "retrieval_prompt_tokens": 0,
        "retrieval_completion_tokens": 0,
        "pre_ce_candidates": len(candidates),
        # Latency decomposition (local retrieval / CE / remote answer).
        "retrieval_ms": round((t_retrieval - t0) * 1000, 1),
        "rerank_ms": round((t_rerank - t_retrieval) * 1000, 1),
        "answer_ms": round((t_answer - t_rerank) * 1000, 1),
        "answer_prompt_tokens": prompt_tok,
        "answer_completion_tokens": completion_tok,
        "answer_reasoning_tokens": reasoning_tok,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sample_questions = [
        "What is the difference between accuracy and precision in measurements?",
        "Explain how a Wheatstone bridge works for measuring unknown resistance.",
    ]

    for q in sample_questions:
        result = hybrid_answer(q)
        print(f"\n{'='*80}")
        print(f"Q: {result['query']}")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks (latency={result['latency_ms']}ms):")
        for ch in result["retrieved_chunks"]:
            print(f"  {ch['chunk_id']}  score={ch['score']:.4f}: {ch['text'][:120]}...")
        print(f"\nA: {result['answer']}")
