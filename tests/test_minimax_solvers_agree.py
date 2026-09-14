"""The cutting-plane minimax fit must agree with the monolithic LP.

`solve_minimax_weights_cuts` replaced `solve_minimax_weights_lp` purely for
speed, so the thing to pin down is that it changes no number anyone reports.
These tests compare the two on the *objective*, never on the weights: the
minimax fit routinely has a flat optimal face (duplicated judges, or a target
reachable by several convex combinations), so two correct solvers may legitimately
return different weight vectors that achieve the same worst-context error.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from epcrc import judge
from epcrc.judge import (
    JudgeResponses,
    solve_minimax_weights,
    solve_minimax_weights_cuts,
    solve_minimax_weights_lp,
    worst_context_error,
)

# The experiments report four decimals, so agreement at 1e-6 is two orders of
# magnitude tighter than anything that reaches the paper.
AGREE = 1e-6


def _random_panel(rng, n_judges, n_contexts, items_per_context):
    """A panel of independent dirichlet draws, contexts of differing sizes."""
    blocks = [
        rng.dirichlet(np.ones(3), size=(n, n_judges))
        for n in items_per_context[:n_contexts]
    ]
    return JudgeResponses(blocks, [f"c{c}" for c in range(n_contexts)])


# --------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_judges,n_contexts", [(3, 1), (4, 2), (6, 3), (9, 4)])
def test_the_two_solvers_find_the_same_optimum(n_judges, n_contexts):
    rng = np.random.default_rng(n_judges * 100 + n_contexts)
    panel = _random_panel(rng, n_judges, n_contexts, [17, 11, 23, 7])

    for target in range(n_judges):
        kept = [j for j in range(n_judges) if j != target]
        lp_error, _ = solve_minimax_weights_lp(panel, target, kept)
        cut_error, _ = solve_minimax_weights_cuts(panel, target, kept)
        assert cut_error == pytest.approx(lp_error, abs=AGREE)


def test_agreement_holds_for_every_subset_size():
    """Small |S| is the regime the greedy pruners spend most of their time in."""
    rng = np.random.default_rng(7)
    panel = _random_panel(rng, 7, 3, [13, 9, 5])

    for m in range(1, 7):
        kept = list(range(1, m + 1))
        lp_error, _ = solve_minimax_weights_lp(panel, 0, kept)
        cut_error, _ = solve_minimax_weights_cuts(panel, 0, kept)
        assert cut_error == pytest.approx(lp_error, abs=AGREE), f"|S|={m}"


def test_agreement_when_the_target_is_an_exact_convex_combination():
    """A zero-error optimum: the cut method must reach it, not stall near it."""
    rng = np.random.default_rng(3)
    blocks, true_w = [], np.array([0.25, 0.35, 0.40])
    for n in (15, 8):
        peers = rng.dirichlet(np.ones(3), size=(n, 3))
        target = np.tensordot(peers, true_w, axes=([1], [0]))
        blocks.append(np.concatenate([target[:, None, :], peers], axis=1))
    panel = JudgeResponses(blocks, ["a", "b"])

    lp_error, _ = solve_minimax_weights_lp(panel, 0, [1, 2, 3])
    cut_error, cut_w = solve_minimax_weights_cuts(panel, 0, [1, 2, 3])

    assert lp_error < 1e-8
    assert cut_error < 1e-6
    assert cut_error == pytest.approx(lp_error, abs=AGREE)
    assert np.allclose(cut_w, true_w, atol=1e-5)


def test_agreement_with_duplicated_judges():
    """A flat optimal face, where the weights may differ but the error may not."""
    rng = np.random.default_rng(11)
    base = rng.dirichlet(np.ones(3), size=(12, 3))
    # Judges 1 and 2 are identical, so any split of their weight is optimal.
    block = np.stack([base[:, 0], base[:, 1], base[:, 1], base[:, 2]], axis=1)
    panel = JudgeResponses([block])

    lp_error, _ = solve_minimax_weights_lp(panel, 0, [1, 2, 3])
    cut_error, _ = solve_minimax_weights_cuts(panel, 0, [1, 2, 3])
    assert cut_error == pytest.approx(lp_error, abs=AGREE)


def test_agreement_when_contexts_conflict_sharply():
    """The case the minimax formulation exists for; both must split the weight."""
    peers = np.array([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]], dtype=float)
    ctx_a = np.concatenate([np.array([[[0.7, 0.2, 0.1]]]), peers], axis=1)
    ctx_b = np.concatenate([np.array([[[0.2, 0.7, 0.1]]]), peers], axis=1)
    panel = JudgeResponses([ctx_a, ctx_b], ["a", "b"])

    lp_error, lp_w = solve_minimax_weights_lp(panel, 0, [1, 2])
    cut_error, cut_w = solve_minimax_weights_cuts(panel, 0, [1, 2])

    assert cut_error == pytest.approx(lp_error, abs=AGREE)
    assert np.allclose(cut_w, lp_w, atol=1e-5)


def test_a_context_with_a_single_item_is_not_drowned_out():
    """Per-context averaging under the cut method, mirroring the LP's rows."""
    rng = np.random.default_rng(5)
    big = rng.dirichlet(np.ones(3), size=(200, 4))
    small = rng.dirichlet(np.ones(3), size=(1, 4))
    panel = JudgeResponses([big, small], ["big", "small"])

    lp_error, _ = solve_minimax_weights_lp(panel, 0, [1, 2, 3])
    cut_error, _ = solve_minimax_weights_cuts(panel, 0, [1, 2, 3])
    assert cut_error == pytest.approx(lp_error, abs=AGREE)


# --------------------------------------------------------------------------
# tie-breaking on a flat optimal face
# --------------------------------------------------------------------------

def _pinned_panel(n_pinned=6, n_free=1):
    """A panel where `n_pinned` contexts pin the maximum and `n_free` are slack.

    A pinned context puts the target on a simplex vertex no convex combination of
    the peers can approach, so ``max_c f_c`` takes the same value for *every* w
    and the worst-context objective says nothing whatsoever about the rest.  In a
    free context the target is the exact midpoint of peers 1 and 2, so the
    mean-error tie-break has a unique right answer there: w = (0.5, 0.5, 0).

    The default is the hardest ratio, and that is the point.  The tie-break
    reaches the objective through the *mean* over contexts, so a lone free
    context contributes only 1/7 of its error gap and needs the strongest
    `_TIE_BREAK`; a panel that is mostly free contexts would pass with a constant
    far too small for the deployed panel.
    """
    peers = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ], dtype=float)
    unreachable = np.array([[1.0, 0.0, 0.0]] * 3)

    free_target = 0.5 * (peers[0] + peers[1])
    free = np.concatenate([free_target[None, :], peers])[None, :, :]
    pinned = np.concatenate([np.array([[0.0, 0.0, 1.0]]), unreachable])[None, :, :]

    return JudgeResponses(
        [pinned] * n_pinned + [free] * n_free,
        [f"pinned{c}" for c in range(n_pinned)] + [f"free{c}" for c in range(n_free)],
    )


@pytest.mark.parametrize("n_pinned,n_free", [(1, 6), (1, 1), (3, 4), (6, 1)])
def test_the_free_contexts_are_still_fitted_when_others_pin_the_maximum(
    n_pinned, n_free
):
    """The regression the tie-break exists to prevent, at every mix of contexts.

    Without it the solver returns an arbitrary point of the optimal face, and the
    per-context errors reported next to the fit are an artefact of LP pivoting
    rather than a fact about the panel.  ``(6, 1)`` is the binding case: it fails
    if `_TIE_BREAK` is dropped by even one order of magnitude.
    """
    panel = _pinned_panel(n_pinned, n_free)
    error, w = solve_minimax_weights_cuts(panel, 0, [1, 2, 3])

    # The pinned contexts are unreachable and fix the objective...
    assert error == pytest.approx(1.0, abs=1e-5)
    # ...but the free contexts are still fitted exactly, not left to chance.
    free_error = worst_context_error(
        JudgeResponses([panel.blocks[-1]], ["free"]), 0, [1, 2, 3], w
    )
    assert free_error == pytest.approx(0.0, abs=1e-5)
    assert np.allclose(w, [0.5, 0.5, 0.0], atol=1e-4)


def test_the_tie_break_weight_is_strong_enough_to_be_seen():
    """`_TIE_BREAK` has to clear the convergence tolerance to do anything.

    Tied weight vectors differ in the regularised objective by only
    ``_TIE_BREAK`` times the mean-error gap.  Once that falls below `tol` the
    solver may stop at any of them and the tie-break silently stops working,
    which is exactly how the original two-stage attempt failed.  The margin is
    verified end to end rather than estimated: the binding panel above must break
    at one tenth of the shipped constant.
    """
    default_tol = inspect.signature(
        solve_minimax_weights_cuts
    ).parameters["tol"].default
    assert judge._TIE_BREAK > default_tol

    panel = _pinned_panel(6, 1)
    weakened = judge._TIE_BREAK / 10
    original = judge._TIE_BREAK
    try:
        judge._TIE_BREAK = weakened
        _, w = solve_minimax_weights_cuts(panel, 0, [1, 2, 3])
    finally:
        judge._TIE_BREAK = original
    assert not np.allclose(w, [0.5, 0.5, 0.0], atol=1e-4)


def test_tie_breaking_never_worsens_the_worst_context_error():
    """The refinement is constrained by the stage-one optimum, not a re-solve."""
    rng = np.random.default_rng(31)
    panel = _random_panel(rng, 7, 4, [16, 10, 5, 21])

    for target in range(7):
        kept = [j for j in range(7) if j != target]
        lp_error, _ = solve_minimax_weights_lp(panel, target, kept)
        cut_error, _ = solve_minimax_weights_cuts(panel, target, kept)
        assert cut_error <= lp_error + AGREE


# --------------------------------------------------------------------------
# the contract the certificate relies on
# --------------------------------------------------------------------------

def test_the_reported_error_is_what_the_returned_weights_achieve():
    """Certificates are only meaningful if the error and the weights match."""
    rng = np.random.default_rng(13)
    panel = _random_panel(rng, 6, 3, [19, 6, 11])

    for target in range(6):
        kept = [j for j in range(6) if j != target]
        error, w = solve_minimax_weights_cuts(panel, target, kept)
        assert error == pytest.approx(
            worst_context_error(panel, target, kept, w), abs=1e-12
        )


def test_the_returned_weights_are_a_simplex_point():
    rng = np.random.default_rng(17)
    panel = _random_panel(rng, 8, 2, [14, 9])

    for target in range(8):
        kept = [j for j in range(8) if j != target]
        _, w = solve_minimax_weights_cuts(panel, target, kept)
        assert w.shape == (len(kept),)
        assert w.min() >= 0.0
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_degenerate_subset_sizes_match_the_lp():
    rng = np.random.default_rng(19)
    panel = _random_panel(rng, 4, 2, [10, 6])

    assert solve_minimax_weights_cuts(panel, 0, [])[0] == float("inf")
    assert solve_minimax_weights_cuts(panel, 0, [])[1].size == 0

    lp_error, lp_w = solve_minimax_weights_lp(panel, 0, [2])
    cut_error, cut_w = solve_minimax_weights_cuts(panel, 0, [2])
    assert cut_error == pytest.approx(lp_error, abs=1e-12)
    assert np.allclose(cut_w, lp_w)


# --------------------------------------------------------------------------
# failure is loud
# --------------------------------------------------------------------------

def test_running_out_of_iterations_raises_rather_than_returning_a_guess():
    """A silently unconverged fit would corrupt every downstream claim."""
    rng = np.random.default_rng(23)
    panel = _random_panel(rng, 6, 3, [21, 13, 8])

    with pytest.raises(RuntimeError, match="did not converge"):
        solve_minimax_weights_cuts(panel, 0, [1, 2, 3, 4, 5], tol=1e-14, max_iter=2)


def test_the_public_entry_point_uses_the_cutting_plane_solver():
    rng = np.random.default_rng(29)
    panel = _random_panel(rng, 5, 2, [12, 7])

    public, _ = solve_minimax_weights(panel, 0, [1, 2, 3, 4])
    cuts, _ = solve_minimax_weights_cuts(panel, 0, [1, 2, 3, 4])
    assert public == pytest.approx(cuts, abs=1e-12)
