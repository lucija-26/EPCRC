"""Tests for E0's minimum-feasible-panel search and its evaluation budget.

The exhaustive search is what makes E0's composition gap trustworthy, and the
budget is what makes it runnable at N = 19.  The risk the budget introduces is
that a *bound* gets read as an exact minimum, so these tests check the honesty of
the reporting as much as the arithmetic.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from epcrc.judge import JudgeCoverageFunctional, JudgeResponses
from experiments.experiment_e0_noncomposability import (
    EXHAUSTIVE_EVAL_BUDGET,
    greedy_feasible_panel,
    min_feasible_panel,
)


def _cov(n_judges=8, n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [rng.dirichlet(np.ones(3), size=(n_items, n_judges)) for _ in range(2)]
    responses = JudgeResponses(blocks, ["a", "b"])
    return JudgeCoverageFunctional(
        responses, responses, [f"J{i:02d}" for i in range(n_judges)]
    )


# --------------------------------------------------------------------------
# the exact search
# --------------------------------------------------------------------------

def test_a_generous_budget_searches_exhaustively():
    cov = _cov()
    out = min_feasible_panel(cov, gamma=0.2, max_evals=10**9)
    assert out.is_exact
    assert out.method == "exhaustive"
    assert out.lower_bound == out.size


def test_the_exact_answer_is_feasible_and_minimal():
    cov = _cov()
    gamma = 0.2
    out = min_feasible_panel(cov, gamma, max_evals=10**9)

    assert out.error <= gamma
    assert len(out.subset) == out.size
    # Nothing smaller can work, which is the property the composition gap needs.
    from itertools import combinations
    for k in range(1, out.size):
        for combo in combinations(range(cov.N), k):
            assert cov.compute_coverage(set(combo))[0] > gamma


def test_a_tolerance_of_zero_still_terminates_on_the_full_panel():
    """The full panel reconstructs itself exactly, so it is always feasible."""
    cov = _cov(n_judges=5)
    out = min_feasible_panel(cov, gamma=0.0, max_evals=10**9)
    assert out.error <= 0.0
    assert out.is_exact


# --------------------------------------------------------------------------
# the budget, and what it does to the reported numbers
# --------------------------------------------------------------------------

def test_exhausting_the_budget_falls_back_to_a_labelled_bound():
    cov = _cov()
    # Enough for the singletons only, so no feasible set can have been found.
    out = min_feasible_panel(cov, gamma=0.2, max_evals=cov.N)

    assert not out.is_exact
    assert out.method == "greedy_bound"
    assert out.lower_bound == 2


def test_the_lower_bound_counts_the_layers_that_actually_finished():
    cov = _cov()
    budget = comb(cov.N, 1) + comb(cov.N, 2)  # layers 1 and 2, not 3
    out = min_feasible_panel(cov, gamma=0.05, max_evals=budget)
    assert out.lower_bound == 3
    assert not out.is_exact


def test_the_certified_lower_bound_is_honest():
    """No panel smaller than `lower_bound` may exist, budget or no budget."""
    cov = _cov()
    gamma = 0.05
    out = min_feasible_panel(cov, gamma, max_evals=comb(cov.N, 1) + comb(cov.N, 2))

    from itertools import combinations
    for k in range(1, out.lower_bound):
        for combo in combinations(range(cov.N), k):
            assert cov.compute_coverage(set(combo))[0] > gamma


def test_the_bound_is_an_upper_bound_on_the_true_minimum():
    cov = _cov()
    gamma = 0.2
    exact = min_feasible_panel(cov, gamma, max_evals=10**9)
    bounded = min_feasible_panel(cov, gamma, max_evals=cov.N)

    assert bounded.size >= exact.size
    assert bounded.lower_bound <= exact.size
    assert bounded.error <= gamma


def test_a_budget_that_stops_where_the_answer_lies_is_still_exact():
    """`is_exact` tracks whether the size is pinned down, not how it was found.

    If backward elimination lands on the very size the finished layers already
    ruled out everything below, the interval has width zero and the answer is
    exact even though enumeration was abandoned.
    """
    cov = _cov()
    gamma = 0.2
    exact = min_feasible_panel(cov, gamma, max_evals=10**9)
    budget = sum(comb(cov.N, k) for k in range(1, exact.size))
    out = min_feasible_panel(cov, gamma, max_evals=budget)

    assert out.lower_bound == exact.size
    if out.size == exact.size:
        assert out.is_exact


def test_the_default_budget_keeps_the_controlled_instances_exhaustive():
    """E0's synthetic instances must not silently degrade to bounds."""
    for n_judges in (8, 9):
        assert sum(comb(n_judges, k) for k in range(1, n_judges + 1)) \
            < EXHAUSTIVE_EVAL_BUDGET


def test_the_default_budget_does_not_pretend_to_handle_the_real_panel():
    """At N = 19 the size-10 layer alone dwarfs any budget worth waiting for."""
    assert comb(19, 10) > EXHAUSTIVE_EVAL_BUDGET


# --------------------------------------------------------------------------
# the greedy fallback on its own
# --------------------------------------------------------------------------

def test_greedy_returns_a_feasible_panel():
    cov = _cov()
    for gamma in (0.0, 0.05, 0.2, 1.0):
        size, subset, error = greedy_feasible_panel(cov, gamma)
        assert error <= max(gamma, 0.0) + 1e-12
        assert len(subset) == size
        assert subset == sorted(set(subset))


def test_greedy_shrinks_as_the_tolerance_loosens():
    cov = _cov()
    sizes = [greedy_feasible_panel(cov, g)[0] for g in (0.0, 0.05, 0.1, 0.3)]
    assert sizes == sorted(sizes, reverse=True)


def test_greedy_keeps_everything_when_nothing_may_be_lost():
    cov = _cov()
    size, subset, error = greedy_feasible_panel(cov, gamma=0.0)
    assert size == cov.N
    assert error == pytest.approx(0.0)
