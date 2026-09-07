"""Tests for three-class judge coverage under worst-context total variation."""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.judge import (
    JudgeCoverageFunctional,
    JudgeResponses,
    solve_minimax_weights,
    total_variation,
    worst_context_error,
)
from epcrc.pruning import BackwardEliminationPruner
from epcrc.synthetic_judges import (
    curved_arc,
    duplicated_extremes,
    sagitta,
    to_simplex,
)


def _block(rows):
    """(n_items, n_judges, 3) block from a nested list of probability rows."""
    return np.array(rows, dtype=float)


def test_tv_equals_max_abs_difference_for_three_outcomes():
    """The identity the LP relies on: for distributions, TV = max_k |p_k - q_k|."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=500)
    q = rng.dirichlet(np.ones(3), size=500)

    assert np.allclose(total_variation(p, q), np.abs(p - q).max(axis=-1))


def test_responses_reject_non_distributions():
    with pytest.raises(ValueError, match="deviate from 1"):
        JudgeResponses([_block([[[0.5, 0.2, 0.1]]])])

    with pytest.raises(ValueError, match="negative"):
        JudgeResponses([_block([[[1.2, -0.2, 0.0]]])])

    with pytest.raises(ValueError, match="disagree"):
        JudgeResponses([
            _block([[[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]]),
            _block([[[0.5, 0.3, 0.2]]]),
        ])


def test_synthetic_panels_are_valid_distributions():
    """The contraction construction must never need clipping."""
    for fit, eval_, _ in (duplicated_extremes(n_items=50), curved_arc(n_items=50)):
        for responses in (fit, eval_):
            for block in responses.blocks:
                assert block.min() >= -1e-12
                assert np.allclose(block.sum(axis=2), 1.0)


def test_exact_convex_combination_is_reconstructed():
    """A judge sitting inside the hull is recovered with the right weights."""
    rng = np.random.default_rng(1)
    peers = rng.dirichlet(np.ones(3), size=(40, 2))          # (items, 2 judges, 3)
    true_w = np.array([0.3, 0.7])
    target = np.tensordot(peers, true_w, axes=([1], [0]))    # (items, 3)

    block = np.concatenate([target[:, None, :], peers], axis=1)
    responses = JudgeResponses([block])

    error, w = solve_minimax_weights(responses, target_idx=0, kept_list=[1, 2])

    assert error < 1e-8
    assert np.allclose(w, true_w, atol=1e-6)


def test_minimax_beats_fitting_a_single_context():
    """Balancing two conflicting contexts must lower the worst-context error.

    Context 0 pulls the target onto peer 1 and context 1 onto peer 2.  Fitting
    either context alone is a disaster in the other; the minimax fit splits the
    weight and halves the damage.
    """
    peers = _block([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]])
    ctx_a = np.concatenate([_block([[[0.7, 0.2, 0.1]]]), peers], axis=1)
    ctx_b = np.concatenate([_block([[[0.2, 0.7, 0.1]]]), peers], axis=1)

    both = JudgeResponses([ctx_a, ctx_b], ["a", "b"])
    only_a = JudgeResponses([ctx_a], ["a"])

    minimax_error, minimax_w = solve_minimax_weights(both, 0, [1, 2])
    _, single_w = solve_minimax_weights(only_a, 0, [1, 2])
    single_error = worst_context_error(both, 0, [1, 2], single_w)

    assert np.allclose(minimax_w, [0.5, 0.5], atol=1e-6)
    assert minimax_error == pytest.approx(0.25, abs=1e-6)
    assert single_error > minimax_error


def test_fitting_coverage_is_monotone_in_the_retained_set():
    """S subset of T implies E_fit(T) <= E_fit(S), on the same fitting data."""
    fit, _, names = curved_arc(n_judges=7, n_items=40, n_contexts=2, seed=3)
    cov = JudgeCoverageFunctional(fit, fit, names)

    smaller = {0, 3, 6}
    larger = {0, 2, 3, 5, 6}

    error_small, _ = cov.compute_coverage(smaller)
    error_large, _ = cov.compute_coverage(larger)

    assert error_large <= error_small + 1e-9


def test_retained_judge_reconstructs_itself_exactly():
    fit, eval_, names = curved_arc(n_judges=5, n_items=30, seed=0)
    cov = JudgeCoverageFunctional(fit, eval_, names)

    cert = cov.compute_certificate(2, {1, 2, 3})

    assert cert.uniqueness == 0.0
    assert np.allclose(cert.weights, [0.0, 1.0, 0.0])


def test_empty_panel_covers_nothing():
    fit, eval_, names = curved_arc(n_judges=4, n_items=20, seed=0)
    cov = JudgeCoverageFunctional(fit, eval_, names)

    assert cov.compute_coverage(set())[0] == float("inf")


def test_duplicated_extremes_is_individually_redundant_but_jointly_not():
    """The controlled C1 construction: everyone removable, three must stay."""
    fit, eval_, names = duplicated_extremes(n_items=60, n_contexts=2, seed=0)
    cov = JudgeCoverageFunctional(fit, eval_, names)
    all_judges = set(range(cov.N))

    loo = [cov.compute_certificate(i, all_judges - {i}).uniqueness for i in range(6)]
    assert max(loo) < 1e-9

    # One representative of each extreme is enough, and any two are not.
    assert cov.compute_coverage({0, 2, 4})[0] < 1e-9
    assert cov.compute_coverage({0, 1})[0] > 0.5

    # Deleting every individually removable judge leaves an empty panel.
    assert cov.compute_coverage(all_judges - set(range(6)))[0] == float("inf")


@pytest.mark.parametrize("n_judges", [5, 9, 17])
def test_arc_curvature_follows_the_inverse_square_law(n_judges):
    """Neighbour reconstruction error on an arc scales like k^-2."""
    radius, arc_span = 0.20, 1.6
    fit, eval_, names = curved_arc(
        n_judges=n_judges, n_items=40, radius=radius, arc_span=arc_span,
        n_contexts=1, seed=0,
    )
    cov = JudgeCoverageFunctional(fit, eval_, names)

    middle = n_judges // 2
    error = cov.compute_certificate(middle, set(range(n_judges)) - {middle}).uniqueness

    # The planar sagitta maps to TV through a fixed affine factor, so the ratio
    # is constant across panel sizes even though the error itself is not.
    assert error / sagitta(radius, arc_span, n_judges) == pytest.approx(0.9, abs=0.1)


def test_pruners_run_unchanged_on_a_judge_panel():
    """The point of matching CoverageFunctional's interface."""
    fit, eval_, names = curved_arc(n_judges=9, n_items=40, n_contexts=2, seed=0)
    cov = JudgeCoverageFunctional(fit, eval_, names)
    gamma = 0.02

    result = BackwardEliminationPruner(cov, gamma).run()

    assert result.coverage <= gamma
    assert 0 < len(result.kept_set) < cov.N
    # The arc endpoints are genuine extremes and cannot be dropped.
    assert {0, 8} <= result.kept_set


def test_barycentric_map_sends_the_corners_to_the_pure_outcomes():
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    assert np.allclose(to_simplex(corners), np.eye(3))
