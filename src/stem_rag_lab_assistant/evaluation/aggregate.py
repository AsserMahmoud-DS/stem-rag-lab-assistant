"""Aggregate results_*.json → comparison.json (overall + per-category + latency).

Pure read/compute/write over the frozen ``results_*.json`` files — no LLM calls.
Regenerates ``comparison.json`` for all evaluated methods. The first question
(Q001) is excluded from the average latency (one-time model-loading overhead).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import (
    ANSWER_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    JUDGE_MODEL_NAME,
    RERANKER_MODEL_NAME,
)

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_COMPARISON_PATH = _EVAL_DIR / "comparison.json"

METHODS = ["naive", "hybrid", "hybrid_graph", "lightrag", "lightrag_hybrid"]
METRICS = (
    "groundedness",
    "answer_relevance",
    "context_relevance",
    "ground_truth_agreement",
)
CATEGORIES = ("cross_document", "multi_hop", "same_document", "lookup", "paraphrase")
_LATENCY_EXCLUDE_QIDS = {"Q001"}


def _load_results(method: str) -> dict[str, Any]:
    path = _EVAL_DIR / f"results_{method}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results)
    sums = {k: 0.0 for k in METRICS}
    for r in results:
        for k in METRICS:
            sums[k] += r["scores"][k]["score"]
    return {k: round(sums[k] / n, 2) for k in METRICS}


def compute_comparison(
    methods: list[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate all methods into the comparison dict and persist it.

    Returns the comparison dict (also written to ``comparison.json``).
    """
    methods = methods or METHODS
    output_path = output_path or _COMPARISON_PATH

    comparison: dict[str, Any] = {
        "eval_date": datetime.now(timezone.utc).isoformat(),
        "answer_model": ANSWER_MODEL_NAME,
        "judge_model": JUDGE_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "overall": {},
        "per_category": {},
    }

    all_by_cat: dict[str, dict[str, list[float]]] = {}

    for method in methods:
        data = _load_results(method)
        results = data["results"]

        metric_means = _mean_metrics(results)
        metric_means["mean"] = round(
            sum(metric_means[k] for k in METRICS) / len(METRICS), 2
        )
        metric_means["avg_chunks"] = round(
            sum(len(r.get("retrieved_chunk_ids", [])) for r in results) / len(results), 1
        )

        latency_records = [
            r for r in results if r["question_id"] not in _LATENCY_EXCLUDE_QIDS
        ]
        metric_means["avg_latency_ms"] = round(
            sum(r["latency_ms"]["method_latency_ms"] for r in latency_records)
            / len(latency_records),
            1,
        )

        comparison["overall"][method] = metric_means

        per_record_means: dict[str, list[float]] = {}
        for r in results:
            m = sum(r["scores"][k]["score"] for k in METRICS) / len(METRICS)
            per_record_means.setdefault(r["category"], []).append(m)
        all_by_cat[method] = per_record_means

    for cat in CATEGORIES:
        entry: dict[str, float] = {}
        for method in methods:
            vals = all_by_cat[method].get(cat, [0.0])
            entry[method] = round(sum(vals) / max(1, len(vals)), 2)
        entry["best"] = max(entry, key=entry.get)
        comparison["per_category"][cat] = entry

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info(
        "comparison.json written for %d methods → %s", len(methods), output_path,
    )
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    comparison = compute_comparison()
    print("\nOverall means:")
    for method, row in comparison["overall"].items():
        print(
            f"  {method:16s} mean={row['mean']:4.2f} "
            f"chunks={row['avg_chunks']:4.1f} latency={row['avg_latency_ms']:6.1f}ms"
        )
