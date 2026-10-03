"""Vanilla LightRAG baseline — throwaway pinned adapter (ONLY module importing lightrag-hku). (P7)"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage

from lightrag import LightRAG, QueryParam
from lightrag.base import EmbeddingFunc

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from stem_rag_lab_assistant.index.reranker import (
    last_rerank_doc_counts,
    lightrag_rerank_func,
    reset_rerank_stats,
)
from stem_rag_lab_assistant.methods.common import (
    answer_usage,
    budget_graph_text,
    format_retrieval_context,
)
from stem_rag_lab_assistant.resources import get_embed_model

load_dotenv()

logger = logging.getLogger(__name__)

_CFG = get_config()

# LightRAG returns its internal chunk ids as ``<our_id>-chunk-NNN``; strip the
# suffix so ids match chunks.json (gold overlap + image attachment).
_CHUNK_SUFFIX_RE = re.compile(r"-chunk-\d+$")

# Per-query retrieval-side LLM accounting (native LightRAG extracts keywords
# with an LLM before graph retrieval). Counters are reset around each query.
_LLM_STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

# Graph context captured by the last retrieval, so lightrag_answer can build a
# matched (entities + relations + chunks) context instead of chunks-only.
_LAST_ENTITIES: list[tuple[str, str]] = []
_LAST_RELATIONS: list[tuple[str, str, str]] = []

# Pre-CE chunk candidate pool of the last retrieval, measured from the rerank
# calls LightRAG made (chunk stage is the largest call; entity/relation stages
# retrieve top_k=40 each). None if no rerank call was recorded.
_LAST_PRE_CE_CANDIDATES: int | None = None


def _normalize_lightrag_chunk_id(cid: str) -> str:
    return _CHUNK_SUFFIX_RE.sub("", cid or "")


def _reset_llm_stats() -> None:
    _LLM_STATS.update(calls=0, prompt_tokens=0, completion_tokens=0)


def _llm_cache_enabled() -> bool:
    """Whether LightRAG's LLM response cache is on (default true).

    Disable (``LIGHTRAG_ENABLE_LLM_CACHE=false``) to force a real keyword-LLM
    call per query, so retrieval-cost comparison isn't masked by a
    warm cache from a previous run.
    """
    return os.getenv("LIGHTRAG_ENABLE_LLM_CACHE", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

# ---- paths ----

_CHUNKS_JSON_PATH = Path(__file__).resolve().parents[3] / "chunks.json"
_LIGHTRAG_WORKING_DIR = Path(__file__).resolve().parents[3] / "lightrag_data"

# ---- singleton state ----

_LIGHTRAG: LightRAG | None = None
_LIGHTRAG_READY: bool = False

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
    # why: the one-time insert re-extracts the whole corpus (rate-limited, can
    # exceed an hour); a short timeout would kill a resumable build mid-way.
    return future.result(timeout=14400)


async def _embedding_func(texts: list[str]) -> np.ndarray:
    """Async wrapper around the shared bge-m3 embedder for LightRAG's EmbeddingFunc."""
    embeddings = get_embed_model().get_text_embedding_batch(texts)
    return np.array(embeddings, dtype=np.float32)


def _build_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=_CFG.embedding_dim,
        max_token_size=8192,
        model_name=_CFG.embedding_model,
        func=_embedding_func,
    )


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> str:
    """Async Groq LLM call for LightRAG's internal extraction and query operations.

    Includes retry-with-backoff for rate limits and transient network/5xx
    failures so they're absorbed before LightRAG's pipeline sees them, enabling
    smooth incremental progress.
    """
    import httpx
    from groq import (
        APIConnectionError,
        APITimeoutError,
        AsyncGroq,
        InternalServerError,
        RateLimitError,
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    # why: a generous connect timeout — transient httpx ConnectTimeouts were
    # failing docs before any retry could run.
    client = AsyncGroq(
        api_key=api_key,
        max_retries=0,  # we handle retries ourselves
        timeout=httpx.Timeout(600.0, connect=30.0),
    )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # LightRAG passes internal params via kwargs — forward OpenAI-compatible ones.
    create_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "stop", "seed", "response_format"):
        if kwargs.get(key) is not None:
            create_kwargs[key] = kwargs[key]
    if kwargs.get("max_tokens") is not None:
        create_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
    # why: LightRAG leaves max_tokens=None (unbounded) — cap output from config.
    create_kwargs.setdefault("max_completion_tokens", _CFG.extraction_max_tokens)
    # why: qwen3 reasons by default; its thinking tokens are billed then discarded.
    create_kwargs.setdefault("reasoning_effort", "none")

    max_retries = 5
    base_delay = 15  # seconds
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=_CFG.extraction_llm_model,
                messages=messages,
                **create_kwargs,
            )
            _LLM_STATS["calls"] += 1
            usage = getattr(response, "usage", None)
            if usage is not None:
                _LLM_STATS["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                _LLM_STATS["completion_tokens"] += (
                    getattr(usage, "completion_tokens", 0) or 0
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
        except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
            # why: transient timeouts/connection/5xx errors must not fail a doc —
            # retry them like rate limits instead of letting LightRAG mark it failed.
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s (attempt %d/%d), waiting %ds ...",
                    type(exc).__name__, attempt + 1, max_retries, delay,
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
        # Fairness parity with the contribution: native LightRAG gets the SAME
        # cross-encoder (its default pipeline expects a rerank model).
        rerank_model_func=lightrag_rerank_func,
        enable_llm_cache=_llm_cache_enabled(),
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

    param = QueryParam(mode="mix", only_need_context=True, chunk_top_k=_CFG.lightrag_chunk_top_k)

    # Reset accounting so we capture THIS query's retrieval-side LLM usage
    # (native LightRAG makes a keyword-extraction LLM call before retrieval).
    _reset_llm_stats()
    reset_rerank_stats()
    result = await rag.aquery_llm(query, param)

    global _LAST_ENTITIES, _LAST_RELATIONS, _LAST_PRE_CE_CANDIDATES
    data = result.get("data", {})
    chunks_data = data.get("chunks", [])
    doc_counts = last_rerank_doc_counts()
    _LAST_PRE_CE_CANDIDATES = max(doc_counts) if doc_counts else None
    _LAST_ENTITIES = [
        (e.get("entity_name", ""), e.get("description", ""))
        for e in data.get("entities", [])
    ]
    _LAST_RELATIONS = [
        (r.get("src_id", ""), r.get("tgt_id", ""), r.get("description", ""))
        for r in data.get("relationships", [])
    ]
    logger.info(
        "LightRAG retrieval: %d chunks for query (first 60 chars): %r",
        len(chunks_data), query[:60],
    )

    output: list[dict[str, Any]] = []
    n = max(len(chunks_data), 1)
    for idx, ch in enumerate(chunks_data):
        chunk_id = _normalize_lightrag_chunk_id(ch.get("chunk_id", ""))
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
    t_retrieval = time.perf_counter()
    # Matched context: native LightRAG's entities/relations + chunks, so the
    # only difference vs the contribution is the retrieval policy.
    graph_text = ""
    if _CFG.lh_include_graph_text:
        graph_text = budget_graph_text(_LAST_ENTITIES, _LAST_RELATIONS)
    context_str = format_retrieval_context(chunks, graph_text)
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
    t_answer = time.perf_counter()
    answer = (
        response.message.content
        if hasattr(response, "message")
        else str(response)
    )
    prompt_tok, completion_tok, reasoning_tok = answer_usage(response)

    latency_ms = (t_answer - t0) * 1000

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
        # Retrieval-side LLM accounting (keyword extraction), captured in
        # _lightrag_retrieve_async via _LLM_STATS.
        "retrieval_llm_calls": _LLM_STATS["calls"],
        "retrieval_prompt_tokens": _LLM_STATS["prompt_tokens"],
        "retrieval_completion_tokens": _LLM_STATS["completion_tokens"],
        # Pre-CE candidate pool (largest rerank call = chunk stage).
        "pre_ce_candidates": _LAST_PRE_CE_CANDIDATES,
        # Latency decomposition: native retrieval (incl. its own rerank) vs answer.
        "retrieval_ms": round((t_retrieval - t0) * 1000, 1),
        "answer_ms": round((t_answer - t_retrieval) * 1000, 1),
        "answer_prompt_tokens": prompt_tok,
        "answer_completion_tokens": completion_tok,
        "answer_reasoning_tokens": reasoning_tok,
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
