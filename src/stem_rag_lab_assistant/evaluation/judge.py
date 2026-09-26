"""Judge — 4 custom Groq feedback functions with retry on rate limits. (P9.2/P9.3)"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import groq
import openai

from stem_rag_lab_assistant.generation.groq_client import get_judge_llm

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to score a single aspect of the system's output on a scale of 1 (poor) to 5 (excellent).

Respond with ONLY a JSON object with exactly two keys:
- "score": integer from 1 to 5
- "reason": one-sentence justification for the score

Do NOT include any other text, markdown, or explanation outside the JSON object."""

_GROUNDEDNESS_PROMPT = """Score how well the ANSWER is grounded in (supported by) the provided CONTEXT.

1 = The answer contains claims that directly contradict the context.
2 = The answer contains claims mostly unsupported by the context.
3 = The answer is partially supported; some claims lack context backing.
4 = The answer is well supported; minor claims may lack explicit backing.
5 = Every claim in the answer is explicitly supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}"""

_ANSWER_RELEVANCE_PROMPT = """Score how well the ANSWER directly addresses the QUESTION.

1 = The answer is completely off-topic or irrelevant.
2 = The answer touches the topic but misses the core question.
3 = The answer is partially relevant but incomplete or includes irrelevant tangents.
4 = The answer is mostly relevant with minor tangential content.
5 = The answer is fully relevant and directly addresses the question.

QUESTION:
{question}

ANSWER:
{answer}"""

_CONTEXT_RELEVANCE_PROMPT = """Score how well the retrieved CONTEXT is relevant to the QUESTION.

1 = The context is entirely unrelated to the question.
2 = The context is mostly off-topic with only tangential connection.
3 = The context is partially relevant; some passages are on-topic, others not.
4 = The context is mostly relevant with minor irrelevant passages.
5 = The context is highly relevant and directly addresses the question.

QUESTION:
{question}

CONTEXT:
{context}"""

_GROUND_TRUTH_AGREEMENT_PROMPT = """Score the semantic agreement between the GENERATED ANSWER and the GOLDEN ANSWER.

1 = The answers are contradictory on key facts.
2 = The answers agree on at most one minor point; major disagreement.
3 = The answers partially agree; some points match, others differ.
4 = The answers are largely consistent; minor differences in phrasing or detail.
5 = The answers are semantically equivalent; the generated answer captures all key points of the golden answer.

GENERATED ANSWER:
{answer}

GOLDEN ANSWER:
{golden_answer}"""


class EvalStoppedError(Exception):
    """Raised when rate limit persists past the timeout — progress saved, can resume."""


def _extract_json(response_text: str) -> dict[str, Any]:
    """Extract a JSON object from a response text that might contain extra wrapping."""
    # Try to find a {...} block
    match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', response_text, re.DOTALL)
    if match:
        import json
        return json.loads(match.group(0))
    # Fallback: try to parse the whole text
    import json
    return json.loads(response_text)


def _call_judge_with_retry(
    messages: list[Any],
    timeout_s: float = 60.0,
) -> str:
    """Call the judge LLM with exponential backoff on rate limits.

    Raises EvalStoppedError if rate limit persists past *timeout_s*.
    """
    judge_llm = get_judge_llm()
    start = time.monotonic()
    delay = 1.0
    attempt = 0

    while True:
        attempt += 1
        try:
            response = judge_llm.chat(messages)
            content = (
                response.message.content
                if hasattr(response, "message")
                else str(response)
            )
            return content

        except groq.RateLimitError as e:
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise EvalStoppedError(
                    f"Rate limit persisted for {elapsed:.1f}s (>{timeout_s}s). "
                    f"Progress saved — re-run to resume."
                ) from e
            logger.warning(
                "Rate limited (attempt %d, %.1fs elapsed). Retrying in %.1fs...",
                attempt, elapsed, delay,
            )
            wait = min(delay, timeout_s - elapsed)
            if wait > 0:
                time.sleep(wait)
            delay = min(delay * 2, 16.0)

        except (groq.APIStatusError, openai.RateLimitError) as e:
            status_code = getattr(e, "status_code", 0)
            if status_code == 429 or isinstance(e, openai.RateLimitError):
                elapsed = time.monotonic() - start
                if elapsed > timeout_s:
                    raise EvalStoppedError(
                        f"Rate limit (429) persisted for {elapsed:.1f}s. "
                        f"Progress saved — re-run to resume."
                    ) from e
                logger.warning(
                    "API 429 / openai rate limit (attempt %d, %.1fs elapsed). Retrying in %.1fs...",
                    attempt, elapsed, delay,
                )
                wait = min(delay, timeout_s - elapsed)
                if wait > 0:
                    time.sleep(wait)
                delay = min(delay * 2, 16.0)
            else:
                raise


def _score_from_response(response_text: str, feedback_name: str) -> dict[str, Any]:
    """Parse score and reason from the judge LLM's response."""
    try:
        data = _extract_json(response_text)
        score = int(data.get("score", 0))
        reason = str(data.get("reason", ""))
    except Exception:
        logger.warning(
            "%s: failed to parse JSON from judge response. Raw: %r",
            feedback_name, response_text[:200],
        )
        # Fallback: try to find a digit
        digits = re.findall(r'\b([1-5])\b', response_text)
        score = int(digits[0]) if digits else 0
        reason = response_text[:500]

    # Clamp score
    score = max(1, min(5, score))
    return {"score": score, "reason": reason}


def judge_groundedness(context: str, answer: str) -> dict[str, Any]:
    """Score how well the answer is supported by the retrieved context (1-5)."""
    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=_GROUNDEDNESS_PROMPT.format(context=context, answer=answer),
        ),
    ]
    response_text = _call_judge_with_retry(messages)
    return _score_from_response(response_text, "groundedness")


def judge_answer_relevance(question: str, answer: str) -> dict[str, Any]:
    """Score how well the answer addresses the question (1-5)."""
    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer),
        ),
    ]
    response_text = _call_judge_with_retry(messages)
    return _score_from_response(response_text, "answer_relevance")


def judge_context_relevance(question: str, context: str) -> dict[str, Any]:
    """Score how relevant the retrieved context is to the question (1-5)."""
    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=_CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context),
        ),
    ]
    response_text = _call_judge_with_retry(messages)
    return _score_from_response(response_text, "context_relevance")


def judge_ground_truth_agreement(answer: str, golden_answer: str) -> dict[str, Any]:
    """Score semantic agreement between generated and golden answer (1-5)."""
    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=_GROUND_TRUTH_AGREEMENT_PROMPT.format(
                answer=answer, golden_answer=golden_answer,
            ),
        ),
    ]
    response_text = _call_judge_with_retry(messages)
    return _score_from_response(response_text, "ground_truth_agreement")
