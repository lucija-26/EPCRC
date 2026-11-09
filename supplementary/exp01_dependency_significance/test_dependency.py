"""Tests target inference-unit handling and the absorbing composition identity."""
import numpy as np

from epcrc.judge import JudgeResponses
from supplementary.exp01_dependency_significance.run import (
    dependency_composition, loss_cube, paired_group_bootstrap,
)


def test_absorbing_weights_and_contextwise_bound():
    W = np.array([[0, .6, .4], [.5, 0, .5], [1, 0, 0.]])
    result = dependency_composition(W, [0, 1])
    assert result["status"] == "contracting"
    assert np.allclose(result["composed_weights"], [[1], [1]])
    rng = np.random.default_rng(7)
    responses = JudgeResponses([rng.dirichlet([1, 1, 1], (20, 3)) for _ in range(2)])
    residual = loss_cube(responses, [0, 1], W[:2]).mean(axis=1)
    composed = loss_cube(responses, [0, 1], np.array([[0, 0, 1], [0, 0, 1]])).mean(axis=1)
    bound = residual @ np.asarray(result["inverse"]).T
    assert np.all(composed <= bound + 1e-12)


def test_closed_dependency_cycle_is_not_inverted():
    W = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0.]])
    result = dependency_composition(W, [0, 1])
    assert result["status"] == "not_contracting"
    assert result["composed_weights"] is None
    assert result["spectral_radius"] == 1


def test_paired_bootstrap_preserves_constant_gap_and_group_unit():
    rng = np.random.default_rng(12)
    a = rng.uniform(.01, .12, (2, 12, 2))
    b = a + .3
    groups = np.repeat(np.arange(4), 3)
    result = paired_group_bootstrap(a, b, groups, .2, repetitions=200, seed=1)
    assert result["n_groups"] == 4
    assert np.allclose(result["overall"]["gap_paired_percentile_interval_descriptive"], [.3, .3])
    assert result["overall"]["noncomposability_supported_by_simultaneous_band"]


def test_bootstrap_is_invariant_to_full_replication_of_each_pair():
    rng = np.random.default_rng(4)
    a = rng.uniform(0, .4, (3, 5, 2))
    b = rng.uniform(0, .6, (3, 5, 2))
    groups = np.array(["a", "a", "b", "c", "c"])
    first = paired_group_bootstrap(a, b, groups, .2, repetitions=100, seed=3)
    second = paired_group_bootstrap(np.repeat(a, 2, axis=1), np.repeat(b, 2, axis=1),
                                    np.repeat(groups, 2), .2, repetitions=100, seed=3)
    assert np.allclose(first["simultaneous_band_radius"], second["simultaneous_band_radius"])
    assert np.allclose(first["overall"]["gap_paired_percentile_interval_descriptive"],
                       second["overall"]["gap_paired_percentile_interval_descriptive"])
