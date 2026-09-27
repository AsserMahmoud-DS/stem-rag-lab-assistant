"""Live TruLens/LiteLLM judge smoke test (hits Groq).

Opt in with ``RUN_LIVE_TESTS=1`` and a valid ``GROQ_API_KEY`` (loaded from
``.env``). Marked ``integration`` and skipped by default so the normal suite
never spends API budget. It validates that all four feedback functions return
the expected 0–1 + reason shape and that a correct answer scores at least as
well as an unrelated one on ground-truth agreement.
"""

from __future__ import annotations

import os
import time

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="set RUN_LIVE_TESTS=1 (and GROQ_API_KEY) to run the live tests",
    ),
]

_QUESTION = "What is the difference between accuracy and precision in measurements?"
_GOLDEN = (
    "Accuracy is how close a measurement is to the true value; precision is how "
    "close repeated measurements are to each other."
)
_CONTEXT = (
    "Accuracy refers to how close a measurement is to the true value. "
    "Precision refers to how close repeated measurements are to each other."
)
_UNRELATED = "A Wheatstone bridge measures an unknown resistance using a galvanometer."


def _assert_result_shape(result: dict) -> None:
    assert isinstance(result, dict)
    assert {"score", "reason"} <= set(result)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0, result
    assert isinstance(result["reason"], str)


def test_four_judges_live() -> None:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    # Lazy import: keep the heavy trulens/litellm import out of default collection.
    from stem_rag_lab_assistant.evaluation.judge import (
        judge_answer_relevance,
        judge_context_relevance,
        judge_ground_truth_agreement,
        judge_groundedness,
    )

    t0 = time.perf_counter()
    groundedness = judge_groundedness(_CONTEXT, _GOLDEN)
    answer_relevance = judge_answer_relevance(_QUESTION, _GOLDEN)
    context_relevance = judge_context_relevance(_QUESTION, _CONTEXT)
    gt_good = judge_ground_truth_agreement(_GOLDEN, _GOLDEN)
    gt_unrelated = judge_ground_truth_agreement(_UNRELATED, _GOLDEN)
    elapsed = time.perf_counter() - t0

    for name, result in (
        ("groundedness", groundedness),
        ("answer_relevance", answer_relevance),
        ("context_relevance", context_relevance),
        ("gt_good", gt_good),
        ("gt_unrelated", gt_unrelated),
    ):
        _assert_result_shape(result)
        print(f"{name}: score={result['score']:.3f} reason={result['reason'][:120]!r}")
    print(f"5 live judge calls in {elapsed:.1f}s")

    assert gt_good["score"] >= gt_unrelated["score"], (gt_good, gt_unrelated)
