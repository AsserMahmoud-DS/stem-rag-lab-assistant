"""Cross-encoder reranker — post-retrieval semantic refinement via BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stem_rag_lab_assistant.config import get_config

logger = logging.getLogger(__name__)

_CFG = get_config()

_RERANKER_SINGLETON: CrossEncoderReranker | None = None


class CrossEncoderReranker:
    """Scores (query, chunk) pairs with a cross-encoder and returns top-N.

    Loads once, re-entrant for all queries in a run.
    """

    def __init__(
        self,
        model_name: str = _CFG.reranker_model,
        top_n: int = _CFG.rerank_top_n,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._top_n = top_n
        self._batch_size = batch_size

        logger.info("Loading cross-encoder: %s (device=%s)", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device).eval()

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score each chunk against the query, return top-N by relevance.

        Args:
            query: user query string.
            chunks: list of {chunk_id, text, score, ...} dicts.
            top_n: override max chunks to return (defaults to self._top_n).

        Returns:
            The top-N chunks, each with ``rerank_score`` added.
        """
        if not chunks:
            return []

        n = top_n or self._top_n
        texts = [ch["text"] for ch in chunks]

        # Batch the forward pass: native LightRAG can hand us 100+ candidates,
        # and a single unbatched forward blows up the 6 GB GPU (and times out
        # LightRAG's rerank worker). Batches keep memory bounded and fast.
        scores_flat: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                pairs = [(query, t) for t in batch]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)
                batch_scores = self._model(**inputs, return_dict=True).logits.squeeze(-1)
                scores_flat.extend(batch_scores.tolist())

        scored: list[dict[str, Any]] = []
        for ch, score in zip(chunks, scores_flat):
            ch = dict(ch)
            ch["rerank_score"] = float(score)
            scored.append(ch)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        kept = scored[:n]

        logger.debug(
            "Rerank: %d candidates → top-%d (scores: %s)",
            len(scored),
            len(kept),
            [f"{c['rerank_score']:.3f}" for c in kept],
        )
        return kept


def get_reranker() -> CrossEncoderReranker:
    """Lazy-init singleton — loads model once, reused across queries."""
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is None:
        _RERANKER_SINGLETON = CrossEncoderReranker()
    return _RERANKER_SINGLETON


def rerank_index_scores(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Score raw document strings, LightRAG-style index/relevance output.

    Returns ``[{"index": i, "relevance_score": score}, ...]`` sorted by score
    descending, where ``i`` indexes into ``documents``. This mirrors the shape
    LightRAG's ``rerank_model_func`` expects, so the native baseline can use
    the *same* cross-encoder as the contribution (fairness parity).
    """
    reranker = get_reranker()
    chunks = [{"chunk_id": str(i), "text": t} for i, t in enumerate(documents)]
    kept = reranker.rerank(query, chunks, top_n=top_n or len(chunks))
    return [
        {"index": int(c["chunk_id"]), "relevance_score": float(c["rerank_score"])}
        for c in kept
    ]


async def lightrag_rerank_func(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Async adapter: our cross-encoder as LightRAG's ``rerank_model_func``."""
    import asyncio

    return await asyncio.to_thread(rerank_index_scores, query, documents, top_n)
