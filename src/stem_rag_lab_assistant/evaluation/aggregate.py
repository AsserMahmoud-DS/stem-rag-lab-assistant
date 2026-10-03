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

from stem_rag_lab_assistant.config import RESULTS_SCHEMA_VERSION, config_snapshot

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_COMPARISON_PATH = _EVAL_DIR / "comparison.json"

# Default aggregated set = the iter-2 parity comparison (mirrors
# run_eval.METHOD_NAMES); the legacy methods stay aggregable via `methods=`.
METHODS = ["naive_ce", "hybrid_ce", "lightrag_hybrid_v2", "lightrag_ce"]
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
        "schema_version": RESULTS_SCHEMA_VERSION,
        "config": config_snapshot(),
        "corpus_hash": None,
        "dataset_hash": None,
        "overall": {},
        "per_category": {},
    }

    all_by_cat: dict[str, dict[str, list[float]]] = {}
    first_data: dict[str, Any] | None = None

    for method in methods:
        data = _load_results(method)
        if first_data is None:
            first_data = data
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

        # Capacity-fallback usage (auditability): recomputed from records, not
        # trusted from the file-level summary block.
        fallbacks = [r.get("answer_fallback") or {} for r in results]
        metric_means["n_answer_fallback"] = sum(
            1 for fb in fallbacks if fb.get("used")
        )
        metric_means["n_answer_fallback_unresolved"] = sum(
            1 for fb in fallbacks if fb.get("used") and not fb.get("resolved")
        )

        # Retrieval-side LLM cost (claim A): native LightRAG reports its
        # keyword-extraction call; the LLM-free methods report 0.
        eff = [r.get("efficiency") or {} for r in results]
        metric_means["avg_retrieval_llm_calls"] = round(
            sum(e.get("retrieval_llm_calls", 0) for e in eff) / len(results), 2
        )
        metric_means["avg_retrieval_prompt_tokens"] = round(
            sum(e.get("retrieval_prompt_tokens", 0) for e in eff) / len(results), 1
        )
        metric_means["avg_retrieval_completion_tokens"] = round(
            sum(e.get("retrieval_completion_tokens", 0) for e in eff) / len(results), 1
        )
        # Pre-CE candidate pool (parity claim evidence): mean over the records
        # that carry it (legacy runs predate the instrumentation).
        pre_ce = [
            e["pre_ce_candidates"]
            for e in eff
            if e.get("pre_ce_candidates") is not None
        ]
        if pre_ce:
            metric_means["avg_pre_ce_candidates"] = round(
                sum(pre_ce) / len(pre_ce), 1
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

    if first_data is not None:
        comparison["corpus_hash"] = first_data.get("corpus_hash")
        comparison["dataset_hash"] = first_data.get("dataset_hash")

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
