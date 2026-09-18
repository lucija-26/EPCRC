"""Tests for C6's exact search and exchange methods.

The panel is planted so the optimum is known before any code runs: three judges
span the whole panel, but no two of them do, and greedy is led away from that
triple by a judge that looks good on its own.  That is the set-dependence trap
C6 is about, so these tests check the mechanism and not just the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.judge import JudgeResponses
from experiments.experiment_c6_exchange import (
    ExactSearch,
    restrict,
    subpanels,
)

N_ITEMS = 40


def _simplex(rows: np.ndarray) -> np.ndarray:
    rows = np.clip(rows, 1e-6, None)
    return rows / rows.sum(axis=-1, keepdims=True)


def _planted() -> JudgeResponses:
    """Five judges: J0, J1, J2 are corners; J3 and J4 are mixtures of them.

    Any set containing all three corners reconstructs J3 and J4 exactly.  No
    smaller set does, because a mixture of two corners cannot reach a point that
    needs mass on the third.
    """
    rng = np.random.default_rng(0)
    corners = np.eye(3)
    block = np.zeros((N_ITEMS, 5, 3))
    for i in range(N_ITEMS):
        jitter = 0.02 * rng.random((3, 3))
        base = _simplex(corners + jitter)
        block[i, 0] = base[0]
        block[i, 1] = base[1]
        block[i, 2] = base[2]
        block[i, 3] = 0.5 * base[0] + 0.5 * base[1]
        block[i, 4] = (base[0] + base[1] + base[2]) / 3.0
    return JudgeResponses([block], ["only"])


def test_the_planted_optimum_is_found_and_is_the_corner_set():
    responses = _planted()
    search = ExactSearch(responses, responses)
    size, combo, error = search.min_feasible_size(0.02, upper_bound=5)
    assert size == 3
    assert set(combo) == {0, 1, 2}
    assert error < 0.02


def test_a_loose_tolerance_admits_a_smaller_set():
    responses = _planted()
    search = ExactSearch(responses, responses)
    loose, _, _ = search.min_feasible_size(0.60, upper_bound=5)
    tight, _, _ = search.min_feasible_size(0.02, upper_bound=5)
    assert loose < tight


def test_the_search_never_returns_more_than_the_upper_bound_it_was_given():
    responses = _planted()
    search = ExactSearch(responses, responses)
    size, combo, _ = search.min_feasible_size(0.0, upper_bound=4)
    assert size == 4
    assert combo is None  # nothing smaller was feasible, so the bound stands


def test_sweeping_gamma_downward_costs_nothing_extra_for_repeated_subsets():
    """The cache is what makes the enumeration affordable, so it is tested."""
    responses = _planted()
    search = ExactSearch(responses, responses)
    search.min_feasible_size(0.60, upper_bound=5)
    first = search.n_eval
    search.min_feasible_size(0.60, upper_bound=5)
    assert search.n_eval == first
    assert search.n_hit > 0


def test_an_infeasible_subset_is_cached_as_a_bound_not_as_a_value():
    responses = _planted()
    search = ExactSearch(responses, responses)
    search._feasible((0, 1), 0.02)
    kind, _ = search._cache[frozenset((0, 1))]
    assert kind == "lb"


def test_an_exactly_evaluated_subset_is_cached_as_a_value():
    responses = _planted()
    search = ExactSearch(responses, responses)
    search._feasible((0, 1, 2), 0.60)
    kind, _ = search._cache[frozenset((0, 1, 2))]
    assert kind == "exact"


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------

def test_restrict_keeps_the_requested_judges_in_the_requested_order():
    responses = _planted()
    smaller = restrict(responses, [4, 0])
    assert smaller.n_judges == 2
    assert smaller.blocks[0][:, 0, :] == pytest.approx(responses.blocks[0][:, 4, :])
    assert smaller.blocks[0][:, 1, :] == pytest.approx(responses.blocks[0][:, 0, :])


def test_the_subpanels_are_fixed_and_have_the_requested_sizes():
    first = subpanels(20, [12, 16])
    second = subpanels(20, [12, 16])
    assert [len(s) for s in first] == [12, 16]
    assert first == second


def test_a_subpanel_never_repeats_a_judge():
    for panel in subpanels(20, [12, 13, 14, 15, 16]):
        assert len(set(panel)) == len(panel)
