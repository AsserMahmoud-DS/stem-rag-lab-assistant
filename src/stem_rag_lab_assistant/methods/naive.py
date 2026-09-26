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
from stem_rag_lab_assistant.index.vector_store import load_vector_store, retrieve

logger = logging.getLogger(__name__)

_CFG = get_config()

_EMBED_MODEL: HuggingFaceEmbedding | None = None
_VECTOR_STORE: SimpleVectorStore | None = None


def _get_embed_model() -> HuggingFaceEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=_CFG.embedding_model,
            # Using GPU for query-time embeddings 
            device="cuda",
        )
    return _EMBED_MODEL


def _get_vector_store() -> SimpleVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = load_vector_store()
    return _VECTOR_STORE


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
    store = vector_store or _get_vector_store()
    emb_model = embed_model or _get_embed_model()
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
