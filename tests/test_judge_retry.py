"""Unit tests for evaluation/retry.py — bounded retries + EvalStoppedError."""

from __future__ import annotations

import pytest

from stem_rag_lab_assistant.evaluation import retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry.time, "sleep", lambda *_args, **_kwargs: None)


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("429 rate limit exceeded")
        return "ok"

    assert retry.run_with_retry(fn, max_attempts=5) == "ok"
    assert calls["n"] == 2


def test_exhaustion_raises_eval_stopped() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise RuntimeError("429 rate limit exceeded")

    with pytest.raises(retry.EvalStoppedError):
        retry.run_with_retry(fn, max_attempts=3)
    assert calls["n"] == 3


def test_non_retryable_stops_immediately() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise RuntimeError("invalid api key")

    with pytest.raises(retry.EvalStoppedError):
        retry.run_with_retry(fn, max_attempts=5)
    assert calls["n"] == 1


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 rate limit", True),
        ("Rate limit reached for requests", True),
        ("Request timed out", True),
        ("Connection error", True),
        ("Service Unavailable 503", True),
        ("invalid api key", False),
        ("BadRequestError: unknown field", False),
    ],
)
def test_is_retryable(message: str, expected: bool) -> None:
    assert retry.is_retryable_error(message) is expected
