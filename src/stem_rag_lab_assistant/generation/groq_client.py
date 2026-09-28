"""Shared Groq client — answer model (gpt-oss-20b) + capacity-fallback variant.

The judge moved to TruLens (LiteLLM → Groq) in `evaluation/judge.py`; only the
answer LLMs live here: the default one, plus a same-model variant with a larger
completion budget used by `evaluation/fallback.py` when an answer is empty.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from llama_index.llms.groq import Groq

from stem_rag_lab_assistant.config import get_config

logger = logging.getLogger(__name__)

_ANSWER_LLM: Optional[Groq] = None
_FALLBACK_ANSWER_LLM: Optional[Groq] = None


def _get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY env var not set. "
            "Export it or put it in .env (loaded by python-dotenv)."
        )
    return key


def get_answer_llm() -> Groq:
    """Return a shared Groq LLM instance for answer synthesis (gpt-oss-20b).

    Lazy-init, singleton across the process.
    """
    global _ANSWER_LLM
    if _ANSWER_LLM is None:
        cfg = get_config()
        _ANSWER_LLM = Groq(model=cfg.answer_model, api_key=_get_api_key())
        logger.info("Answer LLM initialised: %s", cfg.answer_model)
    return _ANSWER_LLM


def get_fallback_answer_llm() -> Groq:
    """Same answer model, higher completion budget (capacity fallback only).

    Used when the default-budget answer comes back empty. Only the completion
    budget (``max_completion_tokens``) differs — reasoning effort is left at the
    model default, so the fallback stays fair across methods.
    """
    global _FALLBACK_ANSWER_LLM
    if _FALLBACK_ANSWER_LLM is None:
        cfg = get_config()
        _FALLBACK_ANSWER_LLM = Groq(
            model=cfg.answer_model,
            api_key=_get_api_key(),
            additional_kwargs={
                "max_completion_tokens": cfg.answer_fallback_max_tokens,
            },
        )
        logger.info(
            "Fallback answer LLM initialised: %s (max_completion_tokens=%d)",
            cfg.answer_model, cfg.answer_fallback_max_tokens,
        )
    return _FALLBACK_ANSWER_LLM
