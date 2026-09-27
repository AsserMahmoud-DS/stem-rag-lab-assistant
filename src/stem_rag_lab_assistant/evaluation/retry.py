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

DEFAULT_MAX_ATTEMPTS = 12
DEFAULT_BASE_DELAY_S = 5.0
DEFAULT_MAX_DELAY_S = 60.0
DEFAULT_MAX_TOTAL_WAIT_S = 600.0

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

# Substrings (lowercased) that mark a fatal, non-retryable failure (bad key).
_FATAL_SIGNATURES = (
    "invalid api key",
    "invalid_api_key",
    "expired_api_key",
    "incorrect api key",
    "authenticationerror",
    "unauthorized",
)


class EvalStoppedError(Exception):
    """Raised when a call fails after retries — progress saved, re-run to resume."""


def is_fatal_error(message: str) -> bool:
    """True for non-retryable failures (e.g. an expired/invalid API key)."""
    low = (message or "").lower()
    return any(sig in low for sig in _FATAL_SIGNATURES)


def is_retryable_error(message: str) -> bool:
    """Minimal retry gate: transient API errors retry; fatal ones stop immediately."""
    if is_fatal_error(message):
        return False
    low = (message or "").lower()
    return any(sig in low for sig in _RETRYABLE_SIGNATURES)


def run_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    max_total_wait_s: float = DEFAULT_MAX_TOTAL_WAIT_S,
    **kwargs: Any,
) -> Any:
    """Call *fn* with patient exponential backoff on retryable errors.

    Rate limits (TPM) are rolling windows: rather than giving up after a few
    seconds, wait up to *max_total_wait_s* (default 10 min) with each delay
    capped at *max_delay_s* (60 s), so a call rides out the window and succeeds.
    Fatal errors (expired/invalid API key) stop immediately, and a TPD ceiling
    (which will not clear within the cap) also stops — so the caller can switch
    keys and resume.

    Raises :class:`EvalStoppedError` on a fatal error or once retries are
    exhausted.
    """
    attempts = 0
    waited = 0.0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — classified by message next
            last_error = e
            if not is_retryable_error(str(e)) or attempt == max_attempts:
                break
            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            if waited + delay > max_total_wait_s:
                logger.warning(
                    "Retry wait cap (%.0fs) reached; stopping after %d attempt(s).",
                    max_total_wait_s, attempt,
                )
                break
            logger.warning(
                "Call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, max_attempts, str(e)[:200], delay,
            )
            time.sleep(delay)
            waited += delay

    raise EvalStoppedError(
        f"Call failed after {attempts} attempt(s) (last: {str(last_error)[:200]}). "
        "Progress saved — switch GROQ_API_KEY and re-run to resume."
    ) from last_error
