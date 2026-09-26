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
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._top_n = top_n

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
        pairs = [(query, t) for t in texts]

        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            scores = self._model(**inputs, return_dict=True).logits.squeeze(-1)

        scored: list[dict[str, Any]] = []
        for ch, score in zip(chunks, scores.tolist()):
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
