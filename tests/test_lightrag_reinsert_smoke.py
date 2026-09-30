"""Live smoke test: re-inserting present chunk ids must not re-extract.

Gated by ``RUN_LIVE_TESTS=1`` (the first insert makes real Groq extraction
calls). Verifies the property we rely on — LightRAG's enqueue filters ids that
already exist in ``doc_status`` (``filter_keys = keys - existing``), so a second
init over the same corpus leaves the graph and processed-doc set unchanged.

Embeddings are stubbed: they are irrelevant to dedup and stubbing avoids loading
bge-m3. The extraction LLM is real, so this stays a genuine end-to-end smoke.
"""

from __future__ import annotations

import asyncio
import json
import os

import networkx as nx
import numpy as np
import pytest
from lightrag.base import EmbeddingFunc

from stem_rag_lab_assistant.methods import lightrag_baseline as lb


def _stub_embedding_func(dim: int) -> EmbeddingFunc:
    async def _fn(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), dim), dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=dim, max_token_size=8192, model_name="stub-embed", func=_fn
    )


def _processed_ids(work_dir) -> set[str]:
    data = json.loads((work_dir / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    return {doc_id for doc_id, rec in data.items() if rec.get("status") == "processed"}


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live smoke (Groq extraction); set RUN_LIVE_TESTS=1 to opt in",
)
def test_reinsert_present_ids_does_not_reextract(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.json"
    work_dir = tmp_path / "lightrag_data"
    chunks_path.write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "d": {
                        "doc_id": "d",
                        "n_chunks": 2,
                        "chunks": [
                            {
                                "chunk_id": "d::ch_00001",
                                "text": "A Wheatstone bridge is balanced when the ratio of its arms is equal.",
                            },
                            {
                                "chunk_id": "d::ch_00002",
                                "text": "A galvanometer serves as the null detector in a bridge circuit.",
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(lb, "_CHUNKS_JSON_PATH", chunks_path)
    monkeypatch.setattr(lb, "_LIGHTRAG_WORKING_DIR", work_dir)
    monkeypatch.setattr(
        lb, "_build_embedding_func", lambda: _stub_embedding_func(lb._CFG.embedding_dim)
    )

    def init_once():
        # Simulate a fresh process.
        lb._LIGHTRAG = None
        lb._LIGHTRAG_READY = False
        rag = asyncio.run(lb._init_lightrag())
        asyncio.run(rag.finalize_storages())
        return rag

    init_once()
    graph_path = work_dir / "graph_chunk_entity_relation.graphml"
    graph_1 = nx.read_graphml(graph_path)
    nodes_1, edges_1 = graph_1.number_of_nodes(), graph_1.number_of_edges()
    assert nodes_1 > 0, "first insert should have extracted entities"
    processed_1 = _processed_ids(work_dir)
    assert processed_1 == {"d::ch_00001", "d::ch_00002"}

    # Second init over the same corpus: nothing should be re-extracted.
    init_once()
    graph_2 = nx.read_graphml(graph_path)
    assert (graph_2.number_of_nodes(), graph_2.number_of_edges()) == (nodes_1, edges_1)
    assert _processed_ids(work_dir) == processed_1
