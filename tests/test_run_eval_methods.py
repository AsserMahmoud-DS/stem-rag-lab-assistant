"""run_eval method registry — the parity set resolves and stays the default.

Import-only smoke (no LLM, no stores, no models): the four budget-parity
methods must resolve to callables, and the default evaluated/aggregated sets
must stay in sync (plans/roadmap.md §5.6).
"""

from __future__ import annotations

from stem_rag_lab_assistant.evaluation.aggregate import METHODS as AGG_METHODS
from stem_rag_lab_assistant.evaluation.run_eval import METHOD_NAMES, _get_answer_func

PARITY_METHODS = ["naive_ce", "hybrid_ce", "lightrag_hybrid_v2", "lightrag_ce"]

# Legacy methods stay registered and runnable via an explicit `methods` arg.
LEGACY_METHODS = ["naive", "hybrid", "lightrag", "lightrag_hybrid"]


def test_default_evaluated_set_is_the_parity_four() -> None:
    assert METHOD_NAMES == PARITY_METHODS


def test_aggregate_default_matches_run_eval() -> None:
    assert AGG_METHODS == METHOD_NAMES


def test_registry_resolves_all_parity_methods() -> None:
    for method in PARITY_METHODS:
        assert callable(_get_answer_func(method))


def test_registry_still_resolves_legacy_methods() -> None:
    for method in LEGACY_METHODS:
        assert callable(_get_answer_func(method))


def test_unknown_method_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        _get_answer_func("no_such_method")
