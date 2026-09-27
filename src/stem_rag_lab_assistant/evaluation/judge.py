"""Judge — TruLens feedback functions over Groq (via LiteLLM), 0–1 scale.

Standalone (no TruLlama instrumentation, no dashboard). Native RAG triad
(``groundedness``, ``answer_relevance``, ``context_relevance``) plus a custom
``ground_truth_agreement`` graded against ``dataset.json`` golden answers.

The public function surface mirrors the previous custom judge, so
``run_eval.py`` stays backend-agnostic. Every function returns
``{"score": float in [0, 1], "reason": str}``. Retries and the stop-and-resume
signal live in ``evaluation/retry.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from trulens.providers.litellm import LiteLLM

from stem_rag_lab_assistant.config import get_config
from stem_rag_lab_assistant.evaluation.retry import run_with_retry

logger = logging.getLogger(__name__)

_PROVIDER: LiteLLM | None = None

_GT_SYSTEM_PROMPT = (
    "You are an expert evaluator for a Retrieval-Augmented Generation (RAG) "
    "system. Compare the candidate answer to the golden answer and rate their "
    "semantic equivalence on a 0-10 scale (0 = contradictory, 5 = partially "
    "correct, 10 = fully equivalent). Ignore phrasing differences; focus on "
    "factual content. Preserve technical notation (formulas, units, variable "
    "names)."
)

_GT_USER_PROMPT = """GOLDEN ANSWER:
{golden_answer}

CANDIDATE ANSWER:
{answer}"""


def _get_provider() -> LiteLLM:
    """Lazy singleton TruLens LiteLLM provider pointed at the Groq judge model.

    ``retries=0`` disables both retry layers TruLens/litellm would otherwise
    add on top of ours: the TruLens endpoint's short backoff (2/4/8 s — far too
    short to ride out a 60 s TPM window) and litellm's SDK retries
    (``litellm.num_retries``). The retry/stop policy is owned by
    ``evaluation/retry.run_with_retry``.
    """
    global _PROVIDER
    if _PROVIDER is None:
        import litellm

        litellm.num_retries = 0
        model = f"groq/{get_config().judge_model}"
        _PROVIDER = LiteLLM(model_engine=model, retries=0)
        logger.info("TruLens LiteLLM judge provider ready: %s", model)
    return _PROVIDER


def _flatten_reasons(reasons: Any) -> str:
    """Render TruLens' reasons payload as a compact string."""
    if not reasons:
        return ""
    if isinstance(reasons, str):
        return reasons[:2000]
    try:
        return json.dumps(reasons, ensure_ascii=False)[:2000]
    except (TypeError, ValueError):
        return str(reasons)[:2000]


def _normalize(result: Any) -> dict[str, Any]:
    """Normalize a TruLens ``(score, reasons)`` (or bare float) to our shape."""
    score, reasons = result if isinstance(result, tuple) else (result, None)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return {"score": max(0.0, min(1.0, score)), "reason": _flatten_reasons(reasons)}


def judge_groundedness(context: str, answer: str) -> dict[str, Any]:
    """Score how well the answer is supported by the retrieved context (0–1)."""
    result = run_with_retry(
        _get_provider().groundedness_measure_with_cot_reasons,
        source=context,
        statement=answer,
    )
    return _normalize(result)


def judge_answer_relevance(question: str, answer: str) -> dict[str, Any]:
    """Score how well the answer addresses the question (0–1)."""
    result = run_with_retry(
        _get_provider().relevance_with_cot_reasons,
        prompt=question,
        response=answer,
    )
    return _normalize(result)


def judge_context_relevance(question: str, context: str) -> dict[str, Any]:
    """Score how relevant the retrieved context is to the question (0–1)."""
    result = run_with_retry(
        _get_provider().context_relevance_with_cot_reasons,
        question=question,
        context=context,
    )
    return _normalize(result)


def judge_ground_truth_agreement(answer: str, golden_answer: str) -> dict[str, Any]:
    """Score semantic agreement between the generated and golden answer (0–1)."""
    result = run_with_retry(
        _get_provider().generate_score_and_reasons,
        system_prompt=_GT_SYSTEM_PROMPT,
        user_prompt=_GT_USER_PROMPT.format(
            golden_answer=golden_answer, answer=answer,
        ),
    )
    return _normalize(result)
