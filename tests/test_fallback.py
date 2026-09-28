"""Unit tests for evaluation/fallback.py — capacity-fallback schema + logic.

No API calls: the LLM entry points are monkeypatched.
"""

from __future__ import annotations

import stem_rag_lab_assistant.evaluation.fallback as fb


def test_is_empty_answer() -> None:
    assert fb.is_empty_answer(None) is True
    assert fb.is_empty_answer("") is True
    assert fb.is_empty_answer("   ") is True
    assert fb.is_empty_answer("[ERROR: model returned no answer]") is True
    assert fb.is_empty_answer("Accuracy is closeness to the true value.") is False


def test_empty_fallback_shape() -> None:
    assert fb.empty_fallback() == {
        "used": False,
        "trigger": None,
        "resolved": None,
        "attempts": None,
        "max_completion_tokens": None,
        "reasoning_effort": None,
    }


def test_fallback_summary_counts() -> None:
    records = [
        {"answer_fallback": fb.empty_fallback()},
        {"answer_fallback": {"used": True, "resolved": True}},
        {"answer_fallback": {"used": True, "resolved": False}},
        {},  # legacy record without the key -> not counted
    ]
    summary = fb.fallback_summary(records)
    assert summary["n_used"] == 2
    assert summary["n_unresolved"] == 1
    assert summary["max_completion_tokens"] == fb.get_config().answer_fallback_max_tokens


def test_attempt_fallback_resolved(monkeypatch) -> None:
    monkeypatch.setattr(fb, "get_fallback_answer_llm", lambda: object())
    monkeypatch.setattr(fb, "_chat_once", lambda context, question, llm: ("An answer.", None))

    answer, meta = fb.attempt_fallback("ctx", "q")
    assert answer == "An answer."
    assert meta["used"] is True
    assert meta["trigger"] == "empty_answer"
    assert meta["resolved"] is True
    assert meta["attempts"] == 1
    assert meta["max_completion_tokens"] == fb.get_config().answer_fallback_max_tokens
    assert meta["reasoning_effort"] == fb._REASONING_EFFORT


def test_attempt_fallback_still_empty(monkeypatch) -> None:
    monkeypatch.setattr(fb, "get_fallback_answer_llm", lambda: object())
    calls = {"n": 0}

    def _empty(context, question, llm):
        calls["n"] += 1
        return "", None

    monkeypatch.setattr(fb, "_chat_once", _empty)
    answer, meta = fb.attempt_fallback("ctx", "q")
    max_attempts = fb.get_config().answer_fallback_max_attempts
    assert answer == ""
    assert meta["used"] is True
    assert meta["resolved"] is False
    assert meta["attempts"] == max_attempts
    assert calls["n"] == max_attempts


def test_attempt_fallback_retries_until_non_empty(monkeypatch) -> None:
    monkeypatch.setattr(fb, "get_fallback_answer_llm", lambda: object())
    calls = {"n": 0}

    def _flaky(context, question, llm):
        calls["n"] += 1
        return ("", None) if calls["n"] == 1 else ("ok", None)

    monkeypatch.setattr(fb, "_chat_once", _flaky)
    answer, meta = fb.attempt_fallback("ctx", "q")
    assert answer == "ok"
    assert meta["resolved"] is True
    assert meta["attempts"] == 2


def test_reasoning_effort_is_not_overridden() -> None:
    """The fallback must record reasoning as unchanged (gpt-oss default)."""
    assert fb._REASONING_EFFORT == "medium"
