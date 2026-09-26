"""Regenerate empty answers using 8B model, re-judge, update results. (one-shot)"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from stem_rag_lab_assistant.evaluation.judge import (
    judge_answer_relevance,
    judge_context_relevance,
    judge_ground_truth_agreement,
    judge_groundedness,
)
from stem_rag_lab_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE

_EVAL_DIR = Path(__file__).resolve().parent


def _get_8b_llm():
    """Create a one-off 8B Groq client (doesn't touch config singleton)."""
    import os
    from llama_index.llms.groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(model="llama-3.1-8b-instant", api_key=key)


def regenerate_method(method: str) -> list[str]:
    """Regenerate all empty answers for one method. Returns list of fixed question_ids."""
    path = _EVAL_DIR / f"results_{method}.json"
    if not path.exists():
        logger.warning("No results file for %s", method)
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    to_fix = [
        r for r in data["results"]
        if not r.get("answer") or r["answer"].startswith("[ERROR")
    ]

    if not to_fix:
        logger.info("%s: no empty answers to fix", method)
        return []

    logger.info("%s: %d empty answers to regenerate", method, len(to_fix))

    llm = _get_8b_llm()
    fixed_ids: list[str] = []

    for r in to_fix:
        qid = r["question_id"]
        question = r["question"]
        context = r.get("context", "")
        golden = r.get("golden_answer", "")

        logger.info("  %s: regenerating with 8B...", qid)

        from llama_index.core.llms import ChatMessage

        messages = [
            ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=ANSWER_USER_TEMPLATE.format(context=context, query=question),
            ),
        ]

        try:
            response = llm.chat(messages)
            answer = (
                response.message.content
                if hasattr(response, "message")
                else str(response)
            )
        except Exception:
            logger.warning("  %s: 8B generation failed, skipping", qid)
            continue

        if answer is None or answer.strip() == "":
            logger.warning("  %s: 8B returned empty, skipping", qid)
            continue

        r["answer"] = answer
        r["answer_model"] = "llama-3.1-8b-instant (regenerated)"

        # Re-judge
        judge_t0 = time.perf_counter()
        try:
            r["scores"]["groundedness"] = judge_groundedness(context, answer)
            r["scores"]["answer_relevance"] = judge_answer_relevance(question, answer)
            r["scores"]["context_relevance"] = judge_context_relevance(question, context)
            r["scores"]["ground_truth_agreement"] = judge_ground_truth_agreement(answer, golden)
        except Exception as e:
            logger.warning("  %s: judging failed (%s), scores kept as-is", qid, e)
        judge_latency = (time.perf_counter() - judge_t0) * 1000

        scores = r["scores"]
        mean = sum(
            scores[k]["score"]
            for k in ("groundedness", "answer_relevance", "context_relevance", "ground_truth_agreement")
        ) / 4.0

        logger.info(
            "  %s: g=%d ar=%d cr=%d gt=%d mean=%.1f (judge=%.0fms)",
            qid,
            scores["groundedness"]["score"],
            scores["answer_relevance"]["score"],
            scores["context_relevance"]["score"],
            scores["ground_truth_agreement"]["score"],
            mean,
            judge_latency,
        )
        fixed_ids.append(qid)

    # Save
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("%s: saved with %d regenerated answers", method, len(fixed_ids))

    return fixed_ids


def regenerate_comparison() -> None:
    """Re-run the comparison after regeneration (delegates to aggregate.py)."""
    from stem_rag_lab_assistant.evaluation.aggregate import compute_comparison

    compute_comparison()
    logger.info("Comparison regenerated → %s", _EVAL_DIR / "comparison.json")


if __name__ == "__main__":
    all_fixed: list[str] = []
    for m in ["naive", "hybrid", "hybrid_graph", "lightrag", "lightrag_hybrid"]:
        fixed = regenerate_method(m)
        all_fixed.extend(fixed)

    if all_fixed:
        regenerate_comparison()
        logger.info("Done. %d answers regenerated across all methods.", len(all_fixed))
    else:
        logger.info("No empty answers found. Nothing to regenerate.")
