"""The oracle tests plan section 17.3 requires of the coverage solver.

Section 17.3 lists six properties the solver must satisfy before any number it
produces may be reported.  Two of them are already pinned in
`tests/test_judge_coverage.py`, because they are statements about the coverage
functional itself rather than about the solver:

  1. a retained judge reconstructs itself with zero numerical error
     -> test_retained_judge_reconstructs_itself_exactly
  2. fitting coverage is non-increasing for nested sets
     -> test_fitting_coverage_is_monotone_in_the_retained_set

The remaining four are here.  They are invariance and correctness properties, so
they are written as exact comparisons wherever the arithmetic allows one, and
the few tolerances that appear are tied to a stated cause.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from epcrc.judge import (
    JudgeCoverageFunctional,
    JudgeResponses,
    solve_minimax_weights,
    solve_minimax_weights_lp,
    worst_context_error,
)
from epcrc.synthetic_judges import curved_arc


# --------------------------------------------------------------------------
# 3. permuting judge columns changes no scalar result after IDs are remapped
# --------------------------------------------------------------------------

def _permute(responses: JudgeResponses, order: np.ndarray) -> JudgeResponses:
    """Reorder the judge axis; `order[new] = old`."""
    return JudgeResponses(
        [block[:, order, :] for block in responses.blocks],
        responses.context_names,
    )


def test_permuting_judge_columns_changes_no_scalar_result():
    """Judge index is a label, so no reported number may depend on it.

    The check is run on the coverage of a set and on each judge's substitution
    error separately, because a solver can be invariant in the aggregate while
    still attributing the error to the wrong judge -- and the per-judge numbers
    are what the backbone and stress experiments read.
    """
    fit, eval_, names = curved_arc(n_judges=7, n_items=40, n_contexts=2, seed=5)
    rng = np.random.default_rng(11)
    order = rng.permutation(7)                      # order[new] = old
    where = np.argsort(order)                       # where[old] = new

    base = JudgeCoverageFunctional(fit, eval_, names)
    shuffled = JudgeCoverageFunctional(
        _permute(fit, order), _permute(eval_, order),
        [names[i] for i in order],
    )

    kept = {0, 2, 5}
    kept_shuffled = {int(where[j]) for j in kept}

    base_error, _ = base.compute_coverage(kept)
    shuffled_error, _ = shuffled.compute_coverage(kept_shuffled)
    assert shuffled_error == pytest.approx(base_error, abs=1e-12)

    for j in range(7):
        a = base.compute_certificate(j, kept)
        b = shuffled.compute_certificate(int(where[j]), kept_shuffled)
        assert b.model_name == a.model_name
        assert b.uniqueness == pytest.approx(a.uniqueness, abs=1e-12)
        # The weights are indexed by position within the sorted kept set, so
        # they have to be compared judge by judge rather than elementwise.
        a_by_judge = dict(zip(sorted(kept), a.weights))
        b_by_judge = {order[j]: w for j, w in zip(sorted(kept_shuffled), b.weights)}
        for judge, weight in a_by_judge.items():
            assert b_by_judge[judge] == pytest.approx(weight, abs=1e-9)


# --------------------------------------------------------------------------
# 4. duplicating an item with half weight changes no result
# --------------------------------------------------------------------------

def _repeat(responses: JudgeResponses, counts) -> JudgeResponses:
    """Repeat row i of every context block `counts[i]` times."""
    return JudgeResponses(
        [np.repeat(block, counts, axis=0) for block in responses.blocks],
        responses.context_names,
    )


def test_duplicating_items_with_half_weight_changes_no_result():
    """The loss depends on the empirical weight of an item, not the row count.

    The code averages over rows and takes no explicit item weights, so splitting
    one item into two copies of half its weight has to be expressed as a change
    of row multiplicities.  Doubling every row is exactly that split applied to
    the whole dataset: each item still carries weight 1/n, now spread over two
    rows.  If anything in the solver were counting rows instead of averaging
    over them, this is where it would show.
    """
    fit, eval_, names = curved_arc(n_judges=6, n_items=35, n_contexts=2, seed=7)
    kept = {0, 3, 5}

    base = JudgeCoverageFunctional(fit, eval_, names)
    doubled = JudgeCoverageFunctional(_repeat(fit, 2), _repeat(eval_, 2), names)

    assert doubled.compute_coverage(kept)[0] == pytest.approx(
        base.compute_coverage(kept)[0], abs=1e-12)


def test_one_duplicated_item_equals_giving_that_item_double_weight():
    """The single-item form of the same property.

    In a dataset of n rows, appending a second copy of item i leaves it with
    weight 2/(n+1).  Doubling every row and then appending two more copies of
    item i gives 2n + 2 rows with item i on 4 of them, which is the same 2/(n+1).
    The two datasets differ in size and in row order but describe the same
    weighted sample, so every reported number has to agree.
    """
    fit, eval_, names = curved_arc(n_judges=6, n_items=25, n_contexts=2, seed=9)
    kept = {1, 2, 4}
    item = 3

    once = np.ones(25, dtype=int)
    once[item] = 2
    twice = np.full(25, 2, dtype=int)
    twice[item] = 4

    a = JudgeCoverageFunctional(_repeat(fit, once), _repeat(eval_, once), names)
    b = JudgeCoverageFunctional(_repeat(fit, twice), _repeat(eval_, twice), names)

    assert b.compute_coverage(kept)[0] == pytest.approx(
        a.compute_coverage(kept)[0], abs=1e-12)
    for j in range(6):
        assert b.compute_certificate(j, kept).uniqueness == pytest.approx(
            a.compute_certificate(j, kept).uniqueness, abs=1e-12)


def test_duplicating_an_item_without_halving_its_weight_does_change_the_result():
    """The invariance above must not be vacuous.

    If the loss were insensitive to item weights altogether then the previous
    two tests would pass for the wrong reason.  Re-weighting one item upward,
    without the compensating split, has to move the number.
    """
    fit, eval_, names = curved_arc(n_judges=6, n_items=25, n_contexts=1, seed=9)
    kept = {1, 2, 4}

    counts = np.ones(25, dtype=int)
    counts[3] = 40
    base = JudgeCoverageFunctional(fit, eval_, names)
    skewed = JudgeCoverageFunctional(_repeat(fit, counts), _repeat(eval_, counts), names)

    assert skewed.compute_coverage(kept)[0] != pytest.approx(
        base.compute_coverage(kept)[0], abs=1e-6)


# --------------------------------------------------------------------------
# 5. LP and brute-force grid search agree on tiny three-judge examples
# --------------------------------------------------------------------------

def _simplex_grid(m: int, steps: int) -> np.ndarray:
    """Every weight vector on the m-simplex whose entries are multiples of 1/steps."""
    rows = []
    for cut in itertools.combinations(range(steps + m - 1), m - 1):
        counts, previous = [], -1
        for c in cut:
            counts.append(c - previous - 1)
            previous = c
        counts.append(steps + m - 2 - previous)
        rows.append(counts)
    return np.asarray(rows, dtype=float) / steps


def _brute_force(responses: JudgeResponses, target: int, kept, steps: int):
    grid = _simplex_grid(len(kept), steps)
    errors = np.array([
        worst_context_error(responses, target, list(kept), w) for w in grid
    ])
    best = int(errors.argmin())
    return float(errors[best]), grid[best]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_lp_matches_brute_force_on_tiny_three_judge_examples(seed):
    """Both solvers must find the same optimum where the optimum can be seen.

    The grid is a lower-resolution search over the same feasible set, so it can
    never beat the LP: the only allowed discrepancy is the LP coming out lower,
    by at most what a 1/200 step can cost.  The objective is a maximum of means
    of total variations, each of which is 1-Lipschitz in the weights under the
    l1 norm, so that gap is bounded by the grid step itself.
    """
    rng = np.random.default_rng(seed)
    blocks = [rng.dirichlet(np.ones(3), size=(12, 4)) for _ in range(2)]
    responses = JudgeResponses(blocks, ["a", "b"])
    kept = [1, 2, 3]
    steps = 200

    lp_error, lp_w = solve_minimax_weights_lp(responses, 0, kept)
    cuts_error, _ = solve_minimax_weights(responses, 0, kept)
    grid_error, _ = _brute_force(responses, 0, kept, steps)

    assert lp_error <= grid_error + 1e-9
    assert lp_error >= grid_error - 1.0 / steps
    assert cuts_error == pytest.approx(lp_error, abs=1e-6)
    # The returned weights must actually achieve the returned error, which is
    # the part a solver can get wrong while still reporting a plausible number.
    assert worst_context_error(responses, 0, kept, lp_w) == pytest.approx(
        lp_error, abs=1e-9)


def test_lp_matches_brute_force_when_the_target_is_inside_the_hull():
    """A case with a known exact answer, so the agreement is not two solvers
    being wrong in the same way."""
    rng = np.random.default_rng(21)
    peers = rng.dirichlet(np.ones(3), size=(15, 3))
    truth = np.array([0.2, 0.5, 0.3])
    target = np.tensordot(peers, truth, axes=([1], [0]))
    block = np.concatenate([target[:, None, :], peers], axis=1)
    responses = JudgeResponses([block])

    lp_error, lp_w = solve_minimax_weights_lp(responses, 0, [1, 2, 3])
    grid_error, _ = _brute_force(responses, 0, [1, 2, 3], steps=100)

    assert lp_error < 1e-8
    assert np.allclose(lp_w, truth, atol=1e-6)
    # Both land on the exact answer here, so the ordering only has to hold up
    # to the rounding that separates two ways of computing the same zero.
    assert grid_error >= lp_error - 1e-12


# --------------------------------------------------------------------------
# 6. the solver detects a deliberately non-reconstructable extreme judge
# --------------------------------------------------------------------------

def test_solver_detects_a_non_reconstructable_extreme_judge():
    """A judge outside the hull of the others must report a large error.

    The peers all sit on the A-versus-B edge with at most 0.1 mass on tie, and
    the planted judge always answers tie.  No convex combination of the peers
    can raise the tie mass above 0.1, so the total variation is at least 0.9
    whatever the weights, and the solver has to say so rather than return a
    small number from a fit that never had a chance.
    """
    rng = np.random.default_rng(3)
    a = rng.uniform(0.0, 0.9, size=(30, 4))
    tie = rng.uniform(0.0, 0.1, size=(30, 4))
    b = 1.0 - a - tie
    peers = np.stack([a, b, tie], axis=-1)
    assert peers.min() >= 0.0

    outlier = np.tile(np.array([0.0, 0.0, 1.0]), (30, 1))
    block = np.concatenate([outlier[:, None, :], peers], axis=1)
    responses = JudgeResponses([block])
    names = ["planted"] + [f"peer_{i}" for i in range(4)]

    error, weights = solve_minimax_weights(responses, 0, [1, 2, 3, 4])

    assert error >= 0.9 - 1e-6
    assert worst_context_error(responses, 0, [1, 2, 3, 4], weights) == pytest.approx(
        error, abs=1e-9)

    # The functional has to inherit that, so the judge cannot be pruned away.
    cov = JudgeCoverageFunctional(responses, responses, names)
    assert cov.compute_coverage({1, 2, 3, 4})[0] >= 0.9 - 1e-6
    assert cov.compute_coverage({0, 1, 2, 3, 4})[0] < 1e-6


def test_the_extreme_judge_is_the_reported_bottleneck():
    """Detecting the error is not enough; it must be attributed to the judge
    that causes it, since that is what selection acts on."""
    rng = np.random.default_rng(4)
    a = rng.uniform(0.0, 0.9, size=(30, 4))
    tie = rng.uniform(0.0, 0.1, size=(30, 4))
    peers = np.stack([a, 1.0 - a - tie, tie], axis=-1)
    outlier = np.tile(np.array([0.0, 0.0, 1.0]), (30, 1))
    block = np.concatenate([outlier[:, None, :], peers], axis=1)
    names = ["planted"] + [f"peer_{i}" for i in range(4)]

    cov = JudgeCoverageFunctional(
        JudgeResponses([block]), JudgeResponses([block]), names)

    _, certs = cov.compute_coverage({1, 2, 3, 4}, return_certificates=True)
    worst = max(certs.values(), key=lambda c: c.uniqueness)
    assert worst.model_name == "planted"
