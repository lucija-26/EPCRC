"""Tests for C7's certification bounds.

A bound is only worth reporting if it is conservative in the direction it
claims to be, so these tests check the direction and the coverage rather than
reproducing the arithmetic.  The coverage test draws from a distribution whose
mean is known, which is the only way to see whether "1 - delta" means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.certification import (
    bernstein_ucb,
    bootstrap_max_error,
    certified_error,
    empirical_error,
    fit_reconstructions,
    group_mean,
    normal_ucb,
    pair_samples,
    per_item_losses,
    worst_test_error,
)
from epcrc.judge import JudgeResponses


def _responses(n_items: int = 60, n_judges: int = 4, seed: int = 0) -> JudgeResponses:
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(2):
        raw = rng.random((n_items, n_judges, 3)) + 0.05
        blocks.append(raw / raw.sum(axis=2, keepdims=True))
    return JudgeResponses(blocks, ["ctx_a", "ctx_b"])


# --------------------------------------------------------------------------
# base items, not pairs
# --------------------------------------------------------------------------

def test_group_mean_averages_within_a_base_item():
    losses = np.array([0.0, 1.0, 0.5, 0.5, 0.2])
    groups = np.array([0, 0, 1, 1, 2])
    assert group_mean(losses, groups) == pytest.approx([0.5, 0.5, 0.2])


def test_group_mean_shrinks_the_sample_to_the_number_of_base_items():
    losses = np.linspace(0, 1, 12)
    groups = np.repeat(np.arange(4), 3)
    assert group_mean(losses, groups).size == 4


# --------------------------------------------------------------------------
# direction of the bounds
# --------------------------------------------------------------------------

def test_every_bound_sits_above_the_sample_mean():
    rng = np.random.default_rng(1)
    sample = rng.beta(2, 5, size=200)
    for bound in (bernstein_ucb, normal_ucb):
        assert bound(sample, 0.05) > sample.mean()


def test_bernstein_is_more_conservative_than_the_normal_approximation():
    """The whole point of E5: the finite-sample bound costs something."""
    rng = np.random.default_rng(2)
    sample = rng.beta(2, 5, size=200)
    assert bernstein_ucb(sample, 0.05) > normal_ucb(sample, 0.05)


def test_a_tighter_delta_gives_a_wider_bound():
    rng = np.random.default_rng(3)
    sample = rng.beta(2, 5, size=200)
    assert bernstein_ucb(sample, 0.01) > bernstein_ucb(sample, 0.10)


def test_a_bound_on_fewer_than_two_points_is_vacuous():
    assert bernstein_ucb(np.array([0.3]), 0.05) == float("inf")
    assert normal_ucb(np.array([]), 0.05) == float("inf")


def test_bernstein_covers_a_known_mean_at_least_as_often_as_nominal():
    """1000 draws from a distribution whose mean we know; delta = 0.05."""
    rng = np.random.default_rng(4)
    true_mean = 2 / 7  # mean of Beta(2, 5)
    misses = sum(
        bernstein_ucb(rng.beta(2, 5, size=40), 0.05) < true_mean for _ in range(1000)
    )
    assert misses == 0  # far more conservative than the nominal 50


# --------------------------------------------------------------------------
# simultaneity
# --------------------------------------------------------------------------

def _samples(n_pairs: int = 6, n_items: int = 50, seed: int = 5):
    rng = np.random.default_rng(seed)
    return {(i, 0): rng.beta(2, 5, size=n_items) for i in range(n_pairs)}


def test_bonferroni_is_wider_than_the_uncorrected_bound():
    samples = _samples()
    corrected = certified_error(samples, 0.05, "bernstein", correct=True)
    uncorrected = certified_error(samples, 0.05, "bernstein", correct=False)
    assert corrected > uncorrected


def test_the_bootstrap_sits_between_uncorrected_and_bonferroni():
    """Simultaneous, but paying for the real dependence instead of the worst case."""
    samples = _samples()
    uncorrected = certified_error(samples, 0.05, "bernstein", correct=False)
    bonferroni = certified_error(samples, 0.05, "bernstein", correct=True)
    boot = bootstrap_max_error(samples, 0.05, n_boot=400, seed=0)
    assert empirical_error(samples) < boot < bonferroni
    assert boot < uncorrected or boot < bonferroni


def test_every_bound_exceeds_the_plain_measured_maximum():
    samples = _samples()
    measured = empirical_error(samples)
    assert certified_error(samples, 0.05, "bernstein", correct=True) > measured
    assert bootstrap_max_error(samples, 0.05, n_boot=400, seed=0) > measured


def test_an_empty_panel_certifies_at_zero():
    assert certified_error({}, 0.05, "bernstein", correct=True) == 0.0
    assert bootstrap_max_error({}, 0.05) == 0.0
    assert empirical_error({}) == 0.0


# --------------------------------------------------------------------------
# the losses being bounded
# --------------------------------------------------------------------------

def test_a_judge_reconstructed_by_itself_has_zero_loss():
    responses = _responses()
    weights = np.array([1.0, 0.0])
    for losses in per_item_losses(responses, 1, [1, 2], weights):
        assert losses == pytest.approx(0.0, abs=1e-12)


def test_losses_stay_inside_the_range_the_bound_assumes():
    responses = _responses()
    _, w = None, np.array([0.5, 0.5])
    for losses in per_item_losses(responses, 0, [1, 2], w):
        assert losses.min() >= 0.0
        assert losses.max() <= 1.0


def test_kept_judges_are_not_certified():
    """Only the reconstructed judges carry risk, so only they are bounded."""
    responses = _responses()
    kept = {0, 1}
    weights = fit_reconstructions(responses, kept, responses.n_judges)
    assert set(weights) == {2, 3}

    groups = np.arange(responses.blocks[0].shape[0])
    samples = pair_samples(responses, groups, kept, weights)
    assert set(samples) == {(2, 0), (2, 1), (3, 0), (3, 1)}


def test_the_full_panel_has_nothing_left_to_go_wrong():
    responses = _responses()
    kept = set(range(responses.n_judges))
    weights = fit_reconstructions(responses, kept, responses.n_judges)
    assert weights == {}
    assert worst_test_error(responses, kept, weights) == 0.0
