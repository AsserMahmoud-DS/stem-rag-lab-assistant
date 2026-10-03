"""RRFRetriever — id-keyed reciprocal-rank fusion (Phase E correctness fix).

The vector and BM25 stores attach slightly different metadata to the same
chunk, so ``node.hash`` differs across retrievers and LlamaIndex's default
fusion cannot merge them. ``RRFRetriever`` keys on ``node_id`` instead, which
both restores cross-retriever boosting and keeps the realised top-k from
shrinking after the post-hoc ``node_id`` dedup.
"""

from __future__ import annotations

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from stem_rag_lab_assistant.index.fusion import RRFRetriever


class _FixedRetriever(BaseRetriever):
    """Returns a fixed node list, standing in for vector/BM25 retrieval."""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        super().__init__()
        self._nodes = nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._nodes


def _nws(nid: str, text: str, metadata: dict, score: float) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, id_=nid, metadata=metadata), score=score
    )


def _make_retriever(retrievers, top_k: int) -> RRFRetriever:
    return RRFRetriever(
        retrievers,
        llm=MockLLM(),  # unused (num_queries=1); avoids resolving Settings.llm
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def test_same_chunk_across_retrievers_is_merged_and_boosted() -> None:
    # Same node_id, different metadata → different hash (documents the bug).
    a_vec = _nws("a", "alpha", {"src": "vector"}, 0.9)
    a_bm = _nws("a", "alpha", {"src": "bm25"}, 0.5)
    assert a_vec.node.hash != a_bm.node.hash

    b = _nws("b", "beta", {}, 0.8)
    c = _nws("c", "gamma", {}, 0.7)

    retriever = _make_retriever(
        [_FixedRetriever([a_vec, b]), _FixedRetriever([a_bm, c])], top_k=3
    )
    out = retriever.retrieve("q")

    ids = [n.node.node_id for n in out]
    assert len(ids) == len(set(ids)) == 3  # "a" merged, not duplicated
    assert set(ids) == {"a", "b", "c"}
    assert ids[0] == "a"  # cross-retriever agreement puts "a" on top
    assert out[0].score > out[1].score


def test_top_k_counts_unique_chunks() -> None:
    a_vec = _nws("a", "alpha", {"src": "vector"}, 0.9)
    a_bm = _nws("a", "alpha", {"src": "bm25"}, 0.5)
    b = _nws("b", "beta", {}, 0.8)
    c = _nws("c", "gamma", {}, 0.7)

    retriever = _make_retriever(
        [_FixedRetriever([a_vec, b]), _FixedRetriever([a_bm, c])], top_k=2
    )
    out = retriever.retrieve("q")

    assert len(out) == 2
    assert len({n.node.node_id for n in out}) == 2  # no duplicate ids in the top-k
