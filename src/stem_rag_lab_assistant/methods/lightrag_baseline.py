"""Vanilla LightRAG baseline — throwaway pinned adapter (ONLY module importing lightrag-hku). (P7)"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from lightrag import LightRAG, QueryParam
from lightrag.base import EmbeddingFunc

from stem_rag_lab_assistant.config import EMBEDDING_MODEL_NAME, EXTRACTION_LLM_MODEL
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE

load_dotenv()

logger = logging.getLogger(__name__)

# ---- paths ----

_CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"
_LIGHTRAG_WORKING_DIR = Path(__file__).resolve().parents[3] / "lightrag_data"

# ---- singleton state ----

_LIGHTRAG: LightRAG | None = None
_LIGHTRAG_READY: bool = False
_EMBED_MODEL: HuggingFaceEmbedding | None = None

# persistent event loop for LightRAG's async API
_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background event loop shared by all LightRAG calls."""
    global _LOOP, _LOOP_THREAD
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        _LOOP_THREAD = threading.Thread(target=_LOOP.run_forever, daemon=True)
        _LOOP_THREAD.start()
        logger.info("Created persistent event loop for LightRAG")
    return _LOOP


def _run_async(coro: Any) -> Any:
    """Bridge: run async coroutine on the persistent loop, return sync result."""
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=3600)


# ---- model singletons ----

def _get_embed_model() -> HuggingFaceEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            device="cuda",
        )
    return _EMBED_MODEL


async def _embedding_func(texts: list[str]) -> np.ndarray:
    """Async wrapper around bge-m3 for LightRAG's EmbeddingFunc."""
    embed_model = _get_embed_model()
    embeddings = embed_model.get_text_embedding_batch(texts)
    return np.array(embeddings, dtype=np.float32)


def _build_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL_NAME,
        func=_embedding_func,
    )


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> str:
    """Async Groq LLM call for LightRAG's internal extraction and query operations.

    Includes retry-with-backoff for rate limits so they're absorbed before
    LightRAG's pipeline sees them, enabling smooth incremental progress.
    """
    from groq import AsyncGroq, RateLimitError

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = AsyncGroq(api_key=api_key, max_retries=0)  # we handle retries ourselves

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # LightRAG passes internal params via kwargs — only forward OpenAI-compatible ones
    create_kwargs: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "stop", "seed"):
        if key in kwargs:
            create_kwargs[key] = kwargs[key]

    max_retries = 5
    base_delay = 15  # seconds
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=EXTRACTION_LLM_MODEL,
                messages=messages,
                **create_kwargs,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %ds ...",
                    attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise


# ---- LightRAG singleton init ----

async def _init_lightrag() -> LightRAG:
    """Initialize LightRAG singleton + insert all chunks (one-time cost)."""
    global _LIGHTRAG, _LIGHTRAG_READY

    if _LIGHTRAG_READY and _LIGHTRAG is not None:
        return _LIGHTRAG

    _LIGHTRAG_WORKING_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(_LIGHTRAG_WORKING_DIR),
        llm_model_func=_llm_model_func,
        embedding_func=_build_embedding_func(),
        llm_model_max_async=1,
        embedding_batch_num=8,
    )

    logger.info("Initializing LightRAG storages at %s ...", _LIGHTRAG_WORKING_DIR)
    await rag.initialize_storages()

    with open(_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    all_texts: list[str] = []
    all_ids: list[str] = []
    for doc in chunks_data.get("docs", {}).values():
        for ch in doc.get("chunks", []):
            all_texts.append(ch["text"])
            all_ids.append(ch["chunk_id"])

    logger.info("Inserting %d chunks into LightRAG (this triggers extraction) ...", len(all_texts))
    t0 = time.perf_counter()
    await rag.ainsert(all_texts, ids=all_ids)
    elapsed = time.perf_counter() - t0
    logger.info("LightRAG insert complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

    _LIGHTRAG = rag
    _LIGHTRAG_READY = True
    return rag


async def _get_lightrag() -> LightRAG:
    return await _init_lightrag()


# ---- retrieval ----

def lightrag_retrieve(query: str) -> list[dict[str, Any]]:
    """Retrieve chunks via LightRAG mix mode — sync wrapper for eval pipeline.

    Returns list of {chunk_id, text, score, metadata} dicts.
    Score = ``rerank_score`` if present in LightRAG's output, else
    position-based descending.
    """
    return _run_async(_lightrag_retrieve_async(query))


async def _lightrag_retrieve_async(query: str) -> list[dict[str, Any]]:
    rag = await _get_lightrag()

    param = QueryParam(mode="mix", only_need_context=True, chunk_top_k=10)

    result = await rag.aquery_llm(query, param)

    chunks_data = result.get("data", {}).get("chunks", [])
    logger.info(
        "LightRAG retrieval: %d chunks for query (first 60 chars): %r",
        len(chunks_data), query[:60],
    )

    output: list[dict[str, Any]] = []
    n = max(len(chunks_data), 1)
    for idx, ch in enumerate(chunks_data):
        chunk_id = ch.get("chunk_id", "")
        text = ch.get("content", "")

        score = ch.get("rerank_score")
        if score is None:
            score = (n - idx) / n

        output.append({
            "chunk_id": chunk_id,
            "text": text,
            "score": float(score),
            "metadata": {},
        })
        logger.debug("  %s  score=%.4f", chunk_id, float(score))

    return output


# ---- answer synthesis ----

def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ch in chunks:
        parts.append(f"[{ch['chunk_id']}]\n{ch['text']}")
    return "\n\n".join(parts)


def lightrag_answer(query: str) -> dict[str, Any]:
    """Full LightRAG baseline: retrieve (mix mode) + shared Groq synthesis.

    Returns dict with keys: query, retrieved_chunks, context, answer,
    latency_ms, model.
    """
    t0 = time.perf_counter()

    chunks = lightrag_retrieve(query)
    context_str = _format_context(chunks)
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
            {"chunk_id": ch["chunk_id"], "text": ch["text"], "score": ch["score"]}
            for ch in chunks
        ],
        "context": context_str,
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
        "model": "lightrag",
    }
    return result


# ---- smoke test ----

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sample_questions = [
        "What is the difference between accuracy and precision in measurements?",
        "Explain how a Wheatstone bridge works for measuring unknown resistance.",
    ]

    for q in sample_questions:
        result = lightrag_answer(q)
        print(f"\n{'=' * 80}")
        print(f"Q: {result['query']}")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks (latency={result['latency_ms']}ms):")
        for ch in result["retrieved_chunks"]:
            print(f"  {ch['chunk_id']}  score={ch['score']:.4f}: {ch['text'][:120]}...")
        print(f"\nA: {result['answer']}")
