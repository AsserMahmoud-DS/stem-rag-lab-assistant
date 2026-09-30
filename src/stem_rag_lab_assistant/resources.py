"""Shared process-wide resource singletons.

One instance per process for the embedder, vector store, BM25 retriever, seed
fusion retriever, reranker, and both graph stores. Every method imports these,
so the 4 evaluated methods (and the dormant ``hybrid_graph``) share identical
in-memory state instead of loading their own copies.

All model/budget knobs come from the shared :class:`RAGConfig` via
``get_config()`` — no literals here.
"""

from __future__ import annotations

import logging

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.generation.groq_client import get_answer_llm
from stem_rag_lab_assistant.index.bm25_store import get_bm25_retriever
from stem_rag_lab_assistant.index.graph_store import GraphStore, load_graph_store
from stem_rag_lab_assistant.index.lightrag_graph_store import (
    LightRAGGraphStore,
    load_lightrag_graph_store,
)
from stem_rag_lab_assistant.index.reranker import get_reranker
from stem_rag_lab_assistant.index.vector_store import (
    VectorRetriever,
    load_or_build_vector_store,
)

__all__ = [
    "get_embed_model",
    "get_vector_store",
    "get_fusion_retriever",
    "get_graph_store",
    "get_lightrag_graph_store",
    "get_reranker",
    "get_bm25_retriever",
]

logger = logging.getLogger(__name__)

_CFG = get_config()

_EMBED_MODEL: HuggingFaceEmbedding | None = None
_VECTOR_STORE: SimpleVectorStore | None = None
_FUSION_RETRIEVER: QueryFusionRetriever | None = None
_GRAPH_STORE: GraphStore | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Shared bge-m3 query embedder (loaded once per process)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=_CFG.embedding_model, device="cuda",
        )
    return _EMBED_MODEL


def get_vector_store() -> SimpleVectorStore:
    """Shared ``SimpleVectorStore`` — loaded if fresh, else rebuilt once per process."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = load_or_build_vector_store()
    return _VECTOR_STORE


def get_fusion_retriever() -> QueryFusionRetriever:
    """Shared RRF(vector + BM25) seed retriever — identical for every method."""
    global _FUSION_RETRIEVER
    if _FUSION_RETRIEVER is not None:
        return _FUSION_RETRIEVER

    vector_retriever = VectorRetriever(
        get_vector_store(), get_embed_model(), top_k=_CFG.vector_top_k,
    )
    _FUSION_RETRIEVER = QueryFusionRetriever(
        [vector_retriever, get_bm25_retriever()],
        similarity_top_k=_CFG.vector_top_k,
        num_queries=1,          # no query generation — RRF only
        mode="reciprocal_rerank",
        use_async=False,
        llm=get_answer_llm(),   # required by constructor, unused with num_queries=1
    )
    logger.info("Shared fusion retriever ready (RRF, top_k=%d)", _CFG.vector_top_k)
    return _FUSION_RETRIEVER


def get_graph_store() -> GraphStore:
    """Shared reader over our own ``graph.json`` (loaded once per process)."""
    global _GRAPH_STORE
    if _GRAPH_STORE is None:
        _GRAPH_STORE = load_graph_store()
    return _GRAPH_STORE


def get_lightrag_graph_store() -> LightRAGGraphStore:
    """Shared reader over LightRAG's persisted graph (singleton in its module)."""
    return load_lightrag_graph_store()
