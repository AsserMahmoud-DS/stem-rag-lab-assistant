"""Tests for LightRAGGraphStore adapter (lightrag_graph_store.py)."""

import os
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stem_rag_lab_assistant.index.lightrag_graph_store import (
    LightRAGGraphStore,
    _normalize_chunk_id,
    load_lightrag_graph_store,
)


def test_chunk_id_normalization():
    """P5.1 — _normalize_chunk_id strips -chunk-NNN suffix; idempotent."""
    assert _normalize_chunk_id("doc::ch_00012-chunk-000") == "doc::ch_00012"
    assert _normalize_chunk_id("doc::ch_00012-chunk-001") == "doc::ch_00012"
    assert _normalize_chunk_id("doc::ch_00012") == "doc::ch_00012"
    assert _normalize_chunk_id("no-suffix") == "no-suffix"


def test_inverted_index_lookup():
    """P5.2 — known chunk_id returns non-empty entity list."""
    store = load_lightrag_graph_store()
    # Known chunk from chap 3 DC&AC BRIDGES
    entities = store.get_entities_for_chunks(["chap 3 DC&AC BRIDGES::ch_00042"])
    assert len(entities) > 0, "Expected entities for known chunk_id"


def test_one_hop_returns_normalized_ids():
    """P5.3 — expand_one_hop yields no -chunk-NNN suffix."""
    store = load_lightrag_graph_store()
    chunks = store.expand_one_hop({"Wheatstone Bridge"})
    assert len(chunks) > 0, "One-hop expansion should return chunk_ids"
    for cid in chunks:
        assert "-chunk-" not in cid, f"Chunk ID still has suffix: {cid}"


def test_relation_expansion():
    """P5.4 — expand_relations returns non-empty for seed entity."""
    store = load_lightrag_graph_store()
    chunks = store.expand_relations({"Wheatstone Bridge"})
    assert len(chunks) > 0, "Relation expansion should return chunk_ids"
    for cid in chunks:
        assert "-chunk-" not in cid, f"Relation chunk ID still has suffix: {cid}"


def test_dedup_against_seeds():
    """P5.5 — expanded pool excludes seed chunk IDs."""
    store = load_lightrag_graph_store()
    entity_chunks = store.expand_one_hop({"Wheatstone Bridge"})
    relation_chunks = store.expand_relations({"Wheatstone Bridge"})
    expanded_pool = entity_chunks | relation_chunks

    # Simulate seeds that overlap with expansion
    fake_seeds = set(list(expanded_pool)[:2])
    new_ids = sorted(expanded_pool - fake_seeds)
    assert all(
        cid not in fake_seeds for cid in new_ids
    ), "Expanded IDs should exclude seeds"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 (needs GPU + GROQ_API_KEY) to run",
)
def test_budget_parity():
    """P5.6 — final context <= 10 chunks (parity with LightRAG chunk_top_k=10)."""
    from stem_rag_lab_assistant.methods.lightrag_hybrid import lightrag_hybrid_answer
    from dotenv import load_dotenv

    load_dotenv()

    result = lightrag_hybrid_answer(
        "What is the difference between accuracy and precision in measurements?"
    )
    chunks = result["retrieved_chunks"]
    assert len(chunks) <= 10, (
        f"Budget exceeded: {len(chunks)} chunks (max 10)"
    )


def test_no_lightrag_hku_import():
    """P5.7 — contribution path does not import lightrag-hku."""
    method_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stem_rag_lab_assistant"
        / "methods"
        / "lightrag_hybrid.py"
    )
    store_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stem_rag_lab_assistant"
        / "index"
        / "lightrag_graph_store.py"
    )

    for path, label in [(method_path, "method"), (store_path, "graph_store")]:
        content = path.read_text()
        assert "import lightrag" not in content, (
            f"'import lightrag' found in {label} file"
        )
        assert "from lightrag" not in content, (
            f"'from lightrag' found in {label} file"
        )


def test_same_config_constants():
    """P5.8 / S2 — budgets come from shared RAGConfig + shared resources."""
    root = Path(__file__).resolve().parents[1] / "src" / "stem_rag_lab_assistant"
    method = (root / "methods" / "lightrag_hybrid.py").read_text()
    resources = (root / "resources.py").read_text()

    # Method reads its own caps from the shared config, not literals
    assert "get_config" in method, "should read the shared RAGConfig"
    assert "_CFG.graph_max_expanded_chunks" in method, (
        "candidate cap should come from RAGConfig.graph_max_expanded_chunks"
    )
    assert "_CFG.rerank_top_n" in method, (
        "reranker should read RAGConfig.rerank_top_n"
    )

    # Seed retriever is the shared process-wide resource
    assert "get_fusion_retriever" in method, "should use shared fusion retriever"

    # The shared retriever reads top-k from config, not a literal
    assert "_CFG.vector_top_k" in resources, (
        "shared fusion retriever should read RAGConfig.vector_top_k"
    )


def test_no_duplicated_resource_singletons():
    """S2 — methods must use resources.py, not their own singleton getters."""
    methods_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stem_rag_lab_assistant"
        / "methods"
    )
    banned = (
        "def _get_embed_model",
        "def _get_fusion_retriever",
        "def _get_reranker",
        "def _get_graph_store",
    )
    for name in (
        "naive.py",
        "hybrid.py",
        "hybrid_graph.py",
        "lightrag_hybrid.py",
        "lightrag_baseline.py",
    ):
        content = (methods_dir / name).read_text()
        for bad in banned:
            assert bad not in content, f"{name} still defines `{bad}`"


def test_entity_name_access_works():
    """Verify that get_entities_for_chunks returns 'name' key for expand_relations."""
    store = load_lightrag_graph_store()
    entities = store.get_entities_for_chunks(["chap 3 DC&AC BRIDGES::ch_00042"])
    assert len(entities) > 0
    for e in entities:
        assert "name" in e, "Entity dict must have 'name' key"
        assert "entity_id" in e, "Entity dict must have 'entity_id' key"
        assert "type" in e, "Entity dict must have 'type' key"
        assert "description" in e, "Entity dict must have 'description' key"
        assert "source_chunk_ids" in e, "Entity dict must have 'source_chunk_ids' key"
