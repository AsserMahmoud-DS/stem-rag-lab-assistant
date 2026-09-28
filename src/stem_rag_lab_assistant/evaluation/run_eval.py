"""Eval runner — run all methods × dataset questions with incremental save + resume.

Judged by the TruLens feedback functions (see ``judge.py``); retries and the
stop-and-resume signal live in ``retry.py``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import config_snapshot
from stem_rag_lab_assistant.evaluation.fallback import (
    SCHEMA_VERSION,
    attempt_fallback,
    empty_fallback,
    fallback_summary,
    is_empty_answer,
)
from stem_rag_lab_assistant.evaluation.judge import (
    judge_answer_relevance,
    judge_context_relevance,
    judge_ground_truth_agreement,
    judge_groundedness,
)
from stem_rag_lab_assistant.evaluation.retry import (
    EvalStoppedError,
    run_with_retry,
)
from stem_rag_lab_assistant.evaluation.sidecar import log_cell
from stem_rag_lab_assistant.methods.common import attach_images

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "questions_dataset" / "dataset.json"
_EVAL_DIR = Path(__file__).resolve().parent

# Evaluated set for the iter-1 re-run. `hybrid_graph` is dormant — kept in code
# and runnable via an explicit CLI arg, but excluded from the default run
# (plans/roadmap.md §2.1).
METHOD_NAMES = ["naive", "hybrid", "lightrag", "lightrag_hybrid"]


def _load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _DATASET_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def _load_results(method: str) -> dict[str, Any]:
    """Load existing results file for a method, or return empty skeleton."""
    path = _EVAL_DIR / f"results_{method}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "method": method,
        "eval_date": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "config": config_snapshot(),
        "results": [],
        "skipped_due_to_rate_limit": [],
    }


def _save_results(method: str, data: dict[str, Any]) -> None:
    path = _EVAL_DIR / f"results_{method}.json"
    data["schema_version"] = SCHEMA_VERSION
    data["answer_fallback"] = fallback_summary(data.get("results", []))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d results for %s → %s", len(data["results"]), method, path)


def _get_completed_ids(data: dict[str, Any]) -> set[str]:
    return {r["question_id"] for r in data.get("results", [])}


def _get_answer_func(method: str):
    if method == "naive":
        from stem_rag_lab_assistant.methods.naive import naive_answer
        return naive_answer
    elif method == "hybrid":
        from stem_rag_lab_assistant.methods.hybrid import hybrid_answer
        return hybrid_answer
    elif method == "hybrid_graph":
        from stem_rag_lab_assistant.methods.hybrid_graph import hybrid_graph_answer
        return hybrid_graph_answer
    elif method == "lightrag":
        from stem_rag_lab_assistant.methods.lightrag_baseline import lightrag_answer
        return lightrag_answer
    elif method == "lightrag_hybrid":
        from stem_rag_lab_assistant.methods.lightrag_hybrid import lightrag_hybrid_answer
        return lightrag_hybrid_answer
    else:
        raise ValueError(f"Unknown method: {method}")


def _run_single_question(
    method: str,
    answer_func,
    q: dict[str, Any],
) -> dict[str, Any]:
    """Run answer + all 4 judge feedbacks for one question.  Raises EvalStoppedError."""
    question = q["question"]
    question_id = q["question_id"]
    golden_answer = q.get("golden_answer", "")
    category = q.get("category", "unknown")

    # 1. Retrieve + answer (patient retry on transient rate limits)
    t0 = time.perf_counter()
    result = run_with_retry(answer_func, question)
    answer = result["answer"]
    retrieved_chunks = result["retrieved_chunks"]
    context = result.get("context", "")
    method_latency = result["latency_ms"]

    # Capacity fallback: if the default-budget answer is empty, retry the same
    # model from the same context with a higher completion budget (reasoning
    # unchanged). Recorded on every record for auditability.
    fallback = empty_fallback()
    if is_empty_answer(answer):
        if answer is None:
            answer = "[ERROR: model returned no answer]"
        fb_answer, fb_meta = attempt_fallback(context, question)
        fallback = fb_meta
        if fb_meta["resolved"]:
            answer = fb_answer

    # 2. Attach images
    images = attach_images(retrieved_chunks, max_images=3)
    attached_image_ids = [img["image_id"] for img in images]

    # 3. Judge (sequential to avoid rate limits)
    judge_t0 = time.perf_counter()
    groundedness = judge_groundedness(context, answer)
    answer_rel = judge_answer_relevance(question, answer)
    context_rel = judge_context_relevance(question, context)
    gt_agreement = judge_ground_truth_agreement(answer, golden_answer)
    judge_latency = (time.perf_counter() - judge_t0) * 1000

    retrieved_ids = [ch["chunk_id"] for ch in retrieved_chunks]

    record: dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "category": category,
        "golden_answer": golden_answer,
        "answer": answer,
        "context": context,
        "retrieved_chunk_ids": retrieved_ids,
        "attached_image_ids": attached_image_ids,
        "answer_fallback": fallback,
        "scores": {
            "groundedness": groundedness,
            "answer_relevance": answer_rel,
            "context_relevance": context_rel,
            "ground_truth_agreement": gt_agreement,
        },
        "latency_ms": _summarize_latency(
            method_latency, judge_latency, len(retrieved_ids),
        ),
        "method": method,
    }

    # 4. Sidecar
    log_cell(
        question_id=question_id,
        method=method,
        retrieved_chunk_ids=retrieved_ids,
        attached_image_ids=attached_image_ids,
        answer=answer,
        latency_ms=method_latency,
    )

    return record


def _summarize_latency(
    method_latency_ms: float,
    judge_latency_ms: float,
    n_chunks: int,
) -> dict[str, Any]:
    return {
        "method_latency_ms": round(method_latency_ms, 1),
        "judge_latency_ms": round(judge_latency_ms, 1),
        "total_latency_ms": round(method_latency_ms + judge_latency_ms, 1),
        "n_retrieved_chunks": n_chunks,
    }


def run_eval(
    dataset: list[dict[str, Any]] | None = None,
    methods: list[str] | None = None,
) -> None:
    """Run all methods × all questions, saving results incrementally.

    Existing results are skipped (resume-safe). Rate limits that persist
    >60s cause immediate stop with progress already saved.
    """
    if dataset is None:
        dataset = _load_dataset()
    if methods is None:
        methods = METHOD_NAMES

    logger.info(
        "Eval run: %d questions × %d methods = %d cells",
        len(dataset), len(methods), len(dataset) * len(methods),
    )

    for method in methods:
        logger.info("=== Method: %s ===", method)
        data = _load_results(method)
        data["config"] = config_snapshot()
        completed = _get_completed_ids(data)
        answer_func = _get_answer_func(method)

        if completed:
            logger.info(
                "Resuming %s: %d/%d already completed",
                method, len(completed), len(dataset),
            )

        for i, q in enumerate(dataset):
            qid = q["question_id"]
            if qid in completed:
                continue

            logger.info(
                "[%s] %s (%s) [%d/%d]",
                method, qid, q.get("category", "?"), i + 1, len(dataset),
            )

            try:
                record = _run_single_question(method, answer_func, q)
            except EvalStoppedError:
                logger.error(
                    "Eval stopped during %s at %s. "
                    "%d/%d results saved for this method. "
                    "Re-run to resume.",
                    method, qid, len(data["results"]), len(dataset),
                )
                sys.exit(1)

            mean_score = sum(
                record["scores"][k]["score"]
                for k in ("groundedness", "answer_relevance", "context_relevance", "ground_truth_agreement")
            ) / 4.0

            logger.info(
                "  → groundedness=%.2f answer_rel=%.2f context_rel=%.2f gt_agree=%.2f | mean=%.2f | latency=%dms",
                record["scores"]["groundedness"]["score"],
                record["scores"]["answer_relevance"]["score"],
                record["scores"]["context_relevance"]["score"],
                record["scores"]["ground_truth_agreement"]["score"],
                mean_score,
                record["latency_ms"]["total_latency_ms"],
            )

            data["results"].append(record)
            _save_results(method, data)

        completed_final = len(data["results"])
        logger.info(
            "%s complete: %d/%d results saved",
            method, completed_final, len(dataset),
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    methods = sys.argv[1:] if len(sys.argv) > 1 else None
    run_eval(methods=methods)
