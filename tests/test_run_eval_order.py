"""run_eval iteration order — question-major (interleaved) + resume.

Server-side answer latency drifts within a run; interleaving the methods per
question keeps the four methods' cells within ~a minute of each other so the
latency comparison is fair. These tests stub out all LLM/storage work.
"""

from __future__ import annotations

from typing import Any

import pytest

import stem_rag_lab_assistant.evaluation.run_eval as re
from stem_rag_lab_assistant.evaluation.run_eval import run_eval

METRICS = (
    "groundedness",
    "answer_relevance",
    "context_relevance",
    "ground_truth_agreement",
)

DATASET = [
    {"question_id": "Q1", "question": "a", "category": "c"},
    {"question_id": "Q2", "question": "b", "category": "c"},
]


def _record(qid: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "scores": {k: {"score": 2.0} for k in METRICS},
        "latency_ms": {"total_latency_ms": 0},
    }


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    store: dict[str, dict[str, Any]] = {}
    calls: list[tuple[str, str]] = []

    def fake_load(method: str) -> dict[str, Any]:
        return store.setdefault(method, {"results": []})

    def fake_save(method: str, data: dict[str, Any]) -> None:
        store[method] = data

    def fake_run(method: str, func: Any, q: dict[str, Any]) -> dict[str, Any]:
        calls.append((q["question_id"], method))
        return _record(q["question_id"])

    monkeypatch.setattr(re, "_load_results", fake_load)
    monkeypatch.setattr(re, "_save_results", fake_save)
    monkeypatch.setattr(re, "_get_answer_func", lambda method: method)
    monkeypatch.setattr(re, "_run_single_question", fake_run)
    monkeypatch.setattr(re, "load_corpus_hash", lambda: "corpus")
    monkeypatch.setattr(re, "dataset_hash", lambda d: "dataset")
    monkeypatch.setattr(re, "config_snapshot", lambda: {})
    return store, calls


def test_interleaves_question_major(harness) -> None:
    _, calls = harness
    run_eval(dataset=DATASET, methods=["m1", "m2"])
    assert calls == [("Q1", "m1"), ("Q1", "m2"), ("Q2", "m1"), ("Q2", "m2")]


def test_resume_skips_completed_cells(harness) -> None:
    store, calls = harness
    store["m1"] = {"results": [_record("Q1")]}
    run_eval(dataset=DATASET, methods=["m1", "m2"])
    # Q1/m1 already done -> skipped; everything else runs, still question-major.
    assert calls == [("Q1", "m2"), ("Q2", "m1"), ("Q2", "m2")]
