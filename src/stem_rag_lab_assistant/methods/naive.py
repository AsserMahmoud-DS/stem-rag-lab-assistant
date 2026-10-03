"""Naive RAG — dense vector retrieval only via SimpleVectorStore top-k. (P1)"""

from __future__ import annotations

import logging
import time
from typing import Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from stem_rag_lab_assistant.index.reranker import get_reranker
from stem_rag_lab_assistant.index.vector_store import retrieve
from stem_rag_lab_assistant.methods.common import answer_usage, format_retrieval_context
from stem_rag_lab_assistant.resources import get_embed_model, get_vector_store

logger = logging.getLogger(__name__)

_CFG = get_config()


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        chunk_id = ch["chunk_id"]
        text = ch["text"]
        parts.append(f"[{chunk_id}]\n{text}")
    return "\n\n".join(parts)


def naive_retrieve(
    query: str,
    vector_store: SimpleVectorStore | None = None,
    embed_model: HuggingFaceEmbedding | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Embed query → retrieve top-k chunks from the vector store.

    Returns list of {chunk_id, text, score, metadata} dicts.
    """
    store = vector_store or get_vector_store()
    emb_model = embed_model or get_embed_model()
    k = top_k or _CFG.vector_top_k

    query_embedding = emb_model.get_query_embedding(query)
    chunks = retrieve(store, query_embedding, top_k=k)

    logger.info("Naive retrieval: %d chunks for query (first 60 chars): %r", len(chunks), query[:60])
    for ch in chunks:
        logger.debug("  %s  score=%.4f", ch["chunk_id"], ch["score"])

    return chunks


def naive_answer(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full naive RAG pipeline: retrieve + synthesise answer.

    Returns dict with keys: query, retrieved_chunks, answer, latency_ms, model.
    """
    t0 = time.perf_counter()

    if chunks is None:
        chunks = naive_retrieve(query)

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
        "model": "naive",
    }
    return result


# ---------------------------------------------------------------------------
# Phase E parity control — naive_ce
# Same candidate budget (lh_pool_cap) and final budget (lh_topn) as the graph
# methods, with the same cross-encoder, but candidates come from plain dense
# retrieval only. Chunks-only context (no graph, no graph text).
# ---------------------------------------------------------------------------


def naive_ce_answer(query: str) -> dict[str, Any]:
    """Budget-parity naive control: vector top-N candidates -> CE -> top-k.

    Differs from the graph methods only in retrieval policy — the candidate
    pool (``lh_pool_cap``), reranker and final budget (``lh_topn``) are
    identical, and context is chunks-only.
    """
    t0 = time.perf_counter()

    candidates = naive_retrieve(query, top_k=_CFG.lh_pool_cap)
    t_retrieval = time.perf_counter()
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
        "model": "naive_ce",
        # Dense retrieval + local cross-encoder: no retrieval-side LLM calls.
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
        result = naive_answer(q)
        print(f"\n{'='*80}")
        print(f"Q: {result['query']}")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks (latency={result['latency_ms']}ms):")
        for ch in result["retrieved_chunks"]:
            print(f"  {ch['chunk_id']}: {ch['text'][:120]}...")
        print(f"\nA: {result['answer']}")
