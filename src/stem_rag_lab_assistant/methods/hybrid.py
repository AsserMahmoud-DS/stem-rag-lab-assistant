"""Hybrid RAG — BM25 + vector fusion via QueryFusionRetriever RRF. (P2)"""

from __future__ import annotations

import logging
import time
from typing import Any

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from stem_rag_lab_assistant.index.bm25_store import get_bm25_retriever
from stem_rag_lab_assistant.index.vector_store import VectorRetriever, load_vector_store

logger = logging.getLogger(__name__)

_CFG = get_config()

_EMBED_MODEL: HuggingFaceEmbedding | None = None
_FUSION_RETRIEVER: QueryFusionRetriever | None = None


def _get_embed_model() -> HuggingFaceEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=_CFG.embedding_model,
            device="cuda",
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
        num_queries=1,              # no query generation — RRF only
        mode="reciprocal_rerank",
        use_async=False,
        llm=get_answer_llm(),       # needed for constructor but not used (num_queries=1)
    )
    logger.info(
        "Hybrid fusion retriever ready (vector_top_k=%d, bm25_top_k=%d, mode=reciprocal_rerank)",
        _CFG.vector_top_k, _CFG.bm25_top_k,
    )
    return _FUSION_RETRIEVER


def hybrid_retrieve(query: str) -> list[dict[str, Any]]:
    """Fuse vector + BM25 via RRF; return [{chunk_id, text, score, metadata}]."""
    retriever = _get_fusion_retriever()
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
