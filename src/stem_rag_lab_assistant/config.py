"""Central configuration.

Two layers:

- **Static paths** — module-level constants (``ROOT_DIR``, ``DATA_DIR``, …).
  Never swept; identity of the corpus location.
- **Tunable knobs** — the frozen :class:`RAGConfig` dataclass, reached through
  :func:`get_config`. Fairness-locked fields must be identical across every
  evaluated method; the rest can be swept with
  ``dataclasses.replace(get_config(), ...)``.
  :func:`config_snapshot` yields an ``asdict`` copy for results artefacts.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Static paths (module constants — not swept)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "dataset"
LOADED_DATA_DIR = ROOT_DIR / "loaded_data"
LIGHTRAG_WORKING_DIR = ROOT_DIR / "lightrag_data"

# Evaluation-results schema version (bump when the results_*.json shape changes).
RESULTS_SCHEMA_VERSION = "iter2"


def to_relative_path(path: Path) -> str:
    """Return ``path`` relative to the project root when possible.

    Artifacts (chunks.json / images.json) store portable paths (e.g.
    ``dataset/Lab1.pdf``) so they are not tied to a machine-specific absolute
    location. Falls back to the raw path if it is outside the project root.
    """
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Tunable knobs (frozen dataclass + get_config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RAGConfig:
    """Frozen snapshot of every tunable knob.

    The *fairness-locked* block must stay identical across all evaluated
    methods (same models + same seed/rerank budget); the remaining fields are
    safe to sweep.
    """

    # --- fairness-locked: models ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    answer_model: str = "openai/gpt-oss-20b"
    judge_model: str = "openai/gpt-oss-120b"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- fairness-locked: seed / rerank budget ---
    # ``vector_top_k`` = dense candidate depth per retriever; for plain
    # ``naive`` it is also the final context size. ``bm25_top_k`` = sparse
    # candidate depth fed into RRF. Both are per-sub-retriever inputs, not the
    # fused count (RRF returns at most ``vector_top_k`` unique chunks).
    vector_top_k: int = 6
    bm25_top_k: int = 6
    # Chunks kept by the cross-encoder when no explicit ``top_n`` is passed.
    # Only the dormant ``hybrid_graph`` path relies on it now; v2 uses
    # ``lh_topn`` and the native baseline passes LightRAG's ``chunk_top_k``.
    rerank_top_n: int = 4
    # Final chunk count native LightRAG keeps after its own retrieval+rerank
    # (passed as ``QueryParam.chunk_top_k``).
    lightrag_chunk_top_k: int = 10

    # --- fairness-locked: answer generation ---
    # Completion budget used only when the first answer is empty (capacity
    # fallback). Reasoning effort is intentionally NOT changed. gpt-oss-20b
    # accepts up to 65,536 output tokens. The fallback retries fresh samples up
    # to ``answer_fallback_max_attempts`` times (Q024 repeatedly hits ``length``).
    answer_fallback_max_tokens: int = 8192
    answer_fallback_max_attempts: int = 4

    # --- chunking (sweepable) ---
    # Character budget per chunk (not tokens): the chunker re-emits the last
    # ``chunk_overlap`` characters of one chunk at the head of the next.
    chunk_size: int = 512
    chunk_overlap: int = 80

    # --- graph retrieval (sweepable) ---
    # One-hop expansion over our own graph.json / LightRAG's graph: each of at
    # most ``graph_seed_entities_cap`` seed entities contributes up to
    # ``graph_neighbour_cap`` neighbouring entities, each yielding up to
    # ``graph_neighbour_cap`` chunks. Used by the dormant ``hybrid_graph`` and
    # ``lightrag_hybrid`` (v1) paths, not by v2.
    graph_neighbour_cap: int = 2
    graph_seed_entities_cap: int = 5
    # Hard cap on new (expanded) chunks returned by those v1 expansion paths.
    graph_max_expanded_chunks: int = 10
    # Unused legacy knob from the iter-1 design; no code reads it.
    graph_expansion_depth: int = 1

    # --- LightRAG-Hybrid v2 (Phase E, sweepable) ---
    # Lexical-bridged graph retrieval:
    #   RRF(BM25 ∪ vector) seeds -> seed-chunk→entity lookup -> one-hop
    #   entity+relation expansion -> cosine pool cap -> full-pool CE rerank ->
    #   top-N -> generation context (+ optional graph text).
    # RRF seed depth (unique chunks steered into the graph); also the fusion
    # top_k. This is the only retrieval-side "search" — no LLM keyword call.
    lh_seed_k: int = 10
    # Max distinct entities pulled from the seed chunks to anchor expansion.
    lh_seed_entity_cap: int = 12
    # Neighbours enumerated per seed entity (and chunks taken per neighbour).
    lh_neighbour_cap: int = 2
    # Cap on the union of seed + expanded chunks handed to the cross-encoder
    # (cosine pre-ranked first). Must exceed ``lh_topn`` for reranking to have
    # any effect; this is the candidate budget P shared with the CE variants.
    lh_pool_cap: int = 28
    # Final chunks kept after CE — v2's generation-time context size.
    lh_topn: int = 10
    # Prepend token-budgeted ENTITIES/RELATIONS descriptions (the entities
    # backing the returned chunks) to the answer context, matching LightRAG's
    # context shape. Also gates graph text for the native baseline.
    lh_include_graph_text: bool = True
    # Shared graph-text ceilings in tokens (enforced by
    # methods.common.budget_graph_text). Defaults mirror LightRAG's native
    # entity/relation budgets (6000/8000) so the baseline is not clipped below
    # its own design.
    lh_max_entity_tokens: int = 6000
    lh_max_relation_tokens: int = 8000
    # Combined entity+relation ceiling; a safety trim only (the per-part caps
    # already sum to this default, so it currently never fires).
    lh_max_graph_tokens: int = 14000

    # --- graph extraction (sweepable) ---
    # One-shot entity/relation extraction from chunk text (our own prompts).
    extraction_llm_model: str = "qwen/qwen3.8-27b"
    # Chunks extracted concurrently (bounded by Groq rate limits).
    extraction_max_concurrent: int = 2
    # Per-chunk prompt caps: distinct entities, and total records
    # (entities + relations) the model may emit.
    extraction_max_entities_per_chunk: int = 30
    extraction_max_total_per_chunk: int = 50
    # Completion-token cap for a single extraction response.
    extraction_max_tokens: int = 4096
    # Retries per chunk on JSON parse failure.
    extraction_retry_max: int = 2

    # --- native LightRAG baseline LLM binding (query-time keyword extraction) ---
    # Serves LightRAG's dual-level (hl/ll) keyword extraction used to search
    # entities/relations at query time. Kept separate from
    # ``extraction_llm_model`` (our own graph.json extraction) so the
    # baseline's runtime binding can change without altering our graph
    # artifacts; inserts are dedup no-ops, so no re-extraction happens.
    lightrag_llm_model: str = "openai/gpt-oss-20b"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _build_config() -> RAGConfig:
    """Build the config from environment overrides, else the field defaults."""
    return RAGConfig(
        embedding_model=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        answer_model=os.getenv("ANSWER_MODEL_NAME", "openai/gpt-oss-20b"),
        judge_model=os.getenv("JUDGE_MODEL_NAME", "openai/gpt-oss-120b"),
        reranker_model=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        vector_top_k=_env_int("VECTOR_TOP_K", 6),
        bm25_top_k=_env_int("BM25_TOP_K", 6),
        rerank_top_n=_env_int("RERANK_TOP_N", 4),
        lightrag_chunk_top_k=_env_int("LIGHTRAG_CHUNK_TOP_K", 10),
        answer_fallback_max_tokens=_env_int("ANSWER_FALLBACK_MAX_TOKENS", 8192),
        answer_fallback_max_attempts=_env_int("ANSWER_FALLBACK_MAX_ATTEMPTS", 4),
        chunk_size=_env_int("CHUNK_SIZE", 512),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 80),
        graph_neighbour_cap=_env_int("GRAPH_NEIGHBOUR_CAP", 2),
        graph_expansion_depth=_env_int("GRAPH_EXPANSION_DEPTH", 1),
        graph_seed_entities_cap=_env_int("GRAPH_SEED_ENTITIES_CAP", 5),
        graph_max_expanded_chunks=_env_int("GRAPH_MAX_EXPANDED_CHUNKS", 10),
        lh_seed_k=_env_int("LH_SEED_K", 10),
        lh_seed_entity_cap=_env_int("LH_SEED_ENTITY_CAP", 12),
        lh_neighbour_cap=_env_int("LH_NEIGHBOUR_CAP", 2),
        lh_pool_cap=_env_int("LH_POOL_CAP", 28),
        lh_topn=_env_int("LH_TOPN", 10),
        lh_include_graph_text=_env_bool("LH_INCLUDE_GRAPH_TEXT", True),
        lh_max_entity_tokens=_env_int("LH_MAX_ENTITY_TOKENS", 6000),
        lh_max_relation_tokens=_env_int("LH_MAX_RELATION_TOKENS", 8000),
        lh_max_graph_tokens=_env_int("LH_MAX_GRAPH_TOKENS", 14000),
        extraction_llm_model=os.getenv(
            "EXTRACTION_LLM_MODEL", "qwen/qwen3.8-27b"
        ),
        extraction_max_concurrent=_env_int("EXTRACTION_MAX_CONCURRENT", 2),
        extraction_max_entities_per_chunk=_env_int(
            "EXTRACTION_MAX_ENTITIES_PER_CHUNK", 30
        ),
        extraction_max_total_per_chunk=_env_int(
            "EXTRACTION_MAX_TOTAL_PER_CHUNK", 50
        ),
        extraction_max_tokens=_env_int("EXTRACTION_MAX_TOKENS", 4096),
        extraction_retry_max=_env_int("EXTRACTION_RETRY_MAX", 2),
        lightrag_llm_model=os.getenv("LIGHTRAG_LLM_MODEL", "openai/gpt-oss-20b"),
    )


_CONFIG: RAGConfig | None = None


def get_config() -> RAGConfig:
    """Return the process-wide frozen config (built once from the environment)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _build_config()
    return _CONFIG


def config_snapshot() -> dict:
    """Serialisable ``asdict`` copy of the config, for results/comparison files."""
    return asdict(get_config())
