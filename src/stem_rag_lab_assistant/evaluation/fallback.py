"""Capacity fallback — re-synthesize empty answers at a higher completion budget.

The first answer attempt uses the default completion budget. If that comes back
empty (``None``/blank/``[ERROR...]``), the *same* answer model is retried with a
larger ``max_completion_tokens`` (``RAGConfig.answer_fallback_max_tokens``) from
the already-retrieved context. Reasoning effort is never changed, so the
fallback only raises the generation budget and stays fair across methods.

This module replaces the deprecated ``_regenerate.py`` (which wrongly swapped
to a different 8B model). It provides:

- ``is_empty_answer`` / ``empty_fallback`` / ``fallback_summary`` — shared schema
  helpers used by ``run_eval`` (live pipeline) and the post-hoc pass below.
- ``attempt_fallback`` — one synthesis retry, no file I/O.
- ``apply_fallback`` — post-hoc repair of existing ``results_*.json`` (adds
  ``schema_version`` + ``answer_fallback`` to every record; retries empties;
  re-judges rescued cells). Idempotent.

Record schema added to every ``results_*.json`` record::

    "answer_fallback": {
        "used": bool,
        "trigger": "empty_answer" | null,
        "resolved": bool | null,          # false if every retry was still empty
        "attempts": int | null,           # synthesis samples tried
        "max_completion_tokens": int | null,
        "reasoning_effort": "medium" | null
    }

Top-level: ``"schema_version": "iter1"`` and an ``"answer_fallback"`` summary.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from llama_index.core.llms import ChatMessage

from stem_rag_lab_assistant.config import RESULTS_SCHEMA_VERSION, get_config
from stem_rag_lab_assistant.evaluation.retry import EvalStoppedError, run_with_retry
from stem_rag_lab_assistant.generation.groq_client import get_fallback_answer_llm
from stem_rag_lab_assistant.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = RESULTS_SCHEMA_VERSION
EVALUATED_METHODS = ["naive", "hybrid", "lightrag", "lightrag_hybrid"]

# gpt-oss-20b's API default reasoning effort (docs). We never override it — the
# fallback only raises the completion budget — so this is recorded purely as the
# fairness evidence that reasoning was unchanged.
_REASONING_EFFORT = "medium"

_EVAL_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def is_empty_answer(answer: Any) -> bool:
    """True if an answer is missing/blank or the sentinel error string."""
    if answer is None:
        return True
    text = str(answer).strip()
    return not text or text.startswith("[ERROR")


def empty_fallback() -> dict[str, Any]:
    """The ``answer_fallback`` stub written on records that never needed it."""
    return {
        "used": False,
        "trigger": None,
        "resolved": None,
        "attempts": None,
        "max_completion_tokens": None,
        "reasoning_effort": None,
    }


def fallback_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-file counts of fallback usage, derived from the records themselves."""
    used = [r for r in records if (r.get("answer_fallback") or {}).get("used")]
    unresolved = [
        r for r in used if not (r.get("answer_fallback") or {}).get("resolved")
    ]
    return {
        "max_completion_tokens": get_config().answer_fallback_max_tokens,
        "n_used": len(used),
        "n_unresolved": len(unresolved),
    }


# ---------------------------------------------------------------------------
# Synthesis (no file I/O)
# ---------------------------------------------------------------------------


def _chat_once(context: str, question: str, llm: Any) -> tuple[str, Any]:
    """One shared-prompt synthesis call; returns (content, raw_response)."""
    messages = [
        ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=ANSWER_USER_TEMPLATE.format(context=context, query=question),
        ),
    ]
    response = llm.chat(messages)
    content = (
        response.message.content if hasattr(response, "message") else str(response)
    )
    return (content or ""), response


def _synthesize_with_retries(
    context: str, question: str, max_attempts: int,
) -> tuple[str, int, Any]:
    """Sample the fallback model up to *max_attempts* times.

    Returns ``(answer, attempts, last_response)``. Each attempt is a fresh
    sample at the higher completion budget; we stop at the first non-empty
    answer. Transient API errors are retried inside each attempt by
    ``run_with_retry`` (exhaustion propagates ``EvalStoppedError``).
    """
    llm = get_fallback_answer_llm()  # config errors (e.g. missing key) propagate
    last_response: Any = None
    for attempt in range(1, max_attempts + 1):
        try:
            answer, last_response = run_with_retry(_chat_once, context, question, llm)
        except EvalStoppedError:
            raise
        except Exception as e:  # noqa: BLE001 — record failure, try next sample
            logger.warning(
                "Capacity fallback attempt %d failed: %s", attempt, str(e)[:200],
            )
            continue
        if not is_empty_answer(answer):
            return answer, attempt, last_response
        logger.info(
            "Capacity fallback attempt %d/%d empty (finish_reason=%r)",
            attempt, max_attempts, _finish_reason(last_response),
        )
    return "", max_attempts, last_response


def attempt_fallback(context: str, question: str) -> tuple[str, dict[str, Any]]:
    """Retry synthesis at the higher completion budget, up to N fresh samples.

    Returns ``(answer, meta)`` where *meta* is the fully-populated
    ``answer_fallback`` object (``used=True``). ``meta["resolved"]`` is False if
    every attempt was still empty. Rate-limit stops propagate as
    ``EvalStoppedError``.
    """
    cfg = get_config()
    meta: dict[str, Any] = {
        "used": True,
        "trigger": "empty_answer",
        "resolved": False,
        "attempts": 0,
        "max_completion_tokens": cfg.answer_fallback_max_tokens,
        "reasoning_effort": _REASONING_EFFORT,
    }
    answer, attempts, _ = _synthesize_with_retries(
        context, question, cfg.answer_fallback_max_attempts,
    )
    meta["attempts"] = attempts
    if not is_empty_answer(answer):
        meta["resolved"] = True
    return answer, meta


# ---------------------------------------------------------------------------
# Post-hoc repair of existing results files
# ---------------------------------------------------------------------------


def _load_results(method: str) -> dict[str, Any]:
    path = _EVAL_DIR / f"results_{method}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_results(method: str, data: dict[str, Any]) -> None:
    path = _EVAL_DIR / f"results_{method}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _rejudge(record: dict[str, Any], answer: str) -> None:
    """Re-run the 4 TruLens feedbacks for a rescued answer and update scores."""
    from stem_rag_lab_assistant.evaluation.judge import (
        judge_answer_relevance,
        judge_context_relevance,
        judge_ground_truth_agreement,
        judge_groundedness,
    )

    question = record["question"]
    context = record.get("context", "")
    golden = record.get("golden_answer", "")
    record["scores"] = {
        "groundedness": judge_groundedness(context, answer),
        "answer_relevance": judge_answer_relevance(question, answer),
        "context_relevance": judge_context_relevance(question, context),
        "ground_truth_agreement": judge_ground_truth_agreement(answer, golden),
    }


def apply_fallback(
    methods: list[str] | None = None,
    *,
    probe_qids: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Normalize schema + rescue empty answers across the given results files.

    Always backfills ``schema_version`` and the ``answer_fallback`` stub onto
    every record (metadata only). Only empty, not-yet-rescued records are
    retried. Saves after each rescued cell (resume-safe). Idempotent.
    """
    methods = methods or EVALUATED_METHODS
    summary: dict[str, Any] = {}

    for method in methods:
        try:
            data = _load_results(method)
        except FileNotFoundError:
            logger.warning("%s: no results file, skipping", method)
            continue
        records = data.get("results", [])
        data["schema_version"] = SCHEMA_VERSION

        targets: list[dict[str, Any]] = []
        for record in records:
            record.setdefault("answer_fallback", empty_fallback())
            fb = record["answer_fallback"]
            if fb.get("used"):
                continue  # already rescued (idempotent)
            if not is_empty_answer(record.get("answer")):
                continue
            if probe_qids is not None and record.get("question_id") not in probe_qids:
                continue
            targets.append(record)

        logger.info("%s: %d empty answers to rescue", method, len(targets))

        for record in targets:
            qid = record.get("question_id")
            if dry_run:
                logger.info(
                    "  %s: would retry at %d tokens (dry-run)",
                    qid, get_config().answer_fallback_max_tokens,
                )
                continue

            context = record.get("context", "")
            question = record.get("question", "")
            answer, meta = attempt_fallback(context, question)
            record["answer_fallback"] = meta

            if meta["resolved"]:
                record["answer"] = answer
                _rejudge(record, answer)
                logger.info("  %s: rescued (fallback resolved)", qid)
            else:
                logger.warning("  %s: still empty after fallback", qid)

            data["answer_fallback"] = fallback_summary(records)
            _save_results(method, data)

        if not dry_run:
            data["answer_fallback"] = fallback_summary(records)
            _save_results(method, data)

        summary[method] = fallback_summary(records)

    return summary


def _finish_reason(response: Any) -> Any:
    raw = getattr(response, "raw", None)
    choices = getattr(raw, "choices", None)
    if choices:
        return getattr(choices[0], "finish_reason", None)
    return None


def probe(
    methods: list[str] | None = None,
    qids: set[str] | None = None,
    limit: int = 10,
) -> None:
    """Diagnostic only: retry empties at the higher budget and print the raw
    result (content length, finish_reason) without modifying any file."""
    methods = methods or EVALUATED_METHODS
    max_attempts = get_config().answer_fallback_max_attempts
    shown = 0
    for method in methods:
        data = _load_results(method)
        for record in data.get("results", []):
            if not is_empty_answer(record.get("answer")):
                continue
            if qids is not None and record.get("question_id") not in qids:
                continue
            if shown >= limit:
                return
            answer, attempts, response = _synthesize_with_retries(
                record.get("context", ""), record.get("question", ""),
                max_attempts,
            )
            print(
                f"[{method}] {record.get('question_id')} "
                f"attempts={attempts}/{max_attempts} "
                f"content_len={len(answer)} "
                f"finish_reason={_finish_reason(response)!r} "
                f"preview={answer[:120]!r}"
            )
            shown += 1


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = sys.argv[1:]
    do_probe = "--probe" in args
    dry_run = "--dry-run" in args
    qids: set[str] | None = None
    for a in args:
        if a.startswith("--qids="):
            qids = {q.strip() for q in a.split("=", 1)[1].split(",") if q.strip()}
    methods = [a for a in args if not a.startswith("-")] or None

    if do_probe:
        probe(methods, qids=qids)
    else:
        result = apply_fallback(methods, probe_qids=qids, dry_run=dry_run)
        print("\nFallback summary:")
        for method, counts in result.items():
            print(f"  {method:16s} {counts}")
