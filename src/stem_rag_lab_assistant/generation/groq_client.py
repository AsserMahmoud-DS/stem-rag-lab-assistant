"""Shared Groq client — answer model (gpt-oss-20b) + judge model (gpt-oss-120b). (P1.1)"""

from __future__ import annotations

import os
import logging
from typing import Optional

from llama_index.llms.groq import Groq

from stem_rag_lab_assistant.config import get_config

logger = logging.getLogger(__name__)

_ANSWER_LLM: Optional[Groq] = None
_JUDGE_LLM: Optional[Groq] = None


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


def get_judge_llm() -> Groq:
    """Return a shared Groq LLM instance for evaluation judging (gpt-oss-120b).

    Lazy-init, singleton across the process.
    """
    global _JUDGE_LLM
    if _JUDGE_LLM is None:
        cfg = get_config()
        _JUDGE_LLM = Groq(model=cfg.judge_model, api_key=_get_api_key())
        logger.info("Judge LLM initialised: %s", cfg.judge_model)
    return _JUDGE_LLM
