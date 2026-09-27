"""Shared retry + stop utilities for the evaluation layer (no heavy imports).

TruLens's endpoint swallows the original exception and re-raises a generic
``RuntimeError``, so callers classify failures by message here. Retries cover
transient API errors; exhaustion (or a clearly fatal error) raises
:class:`EvalStoppedError`, which ``run_eval`` treats as "progress saved — switch
key and re-run to resume".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_S = 2.0

# Substrings (lowercased) that mark a transient, retryable failure.
_RETRYABLE_SIGNATURES = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    " 502",
    " 503",
    " 504",
)


class EvalStoppedError(Exception):
    """Raised when a call fails after retries — progress saved, re-run to resume."""


def is_retryable_error(message: str) -> bool:
    """Minimal retry gate: transient API errors retry; fatal ones stop immediately."""
    low = (message or "").lower()
    return any(sig in low for sig in _RETRYABLE_SIGNATURES)


def run_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    **kwargs: Any,
) -> Any:
    """Call *fn* with bounded exponential backoff on retryable errors.

    Raises :class:`EvalStoppedError` on a non-retryable error or once
    *max_attempts* is reached.
    """
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — classified by message next
            last_error = e
            if not is_retryable_error(str(e)) or attempt == max_attempts:
                break
            delay = base_delay_s * (2 ** (attempt - 1))
            logger.warning(
                "Call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, max_attempts, str(e)[:200], delay,
            )
            time.sleep(delay)

    raise EvalStoppedError(
        f"Call failed after {attempts} attempt(s) (last: {str(last_error)[:200]}). "
        "Progress saved — switch GROQ_API_KEY and re-run to resume."
    ) from last_error
