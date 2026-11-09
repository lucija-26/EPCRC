"""Tests for the E9 joint-removal audit (epcrc/joint_removal.py)."""

from __future__ import annotations

import numpy as np
import pytest

from epcrc import joint_removal as jr
from epcrc.judge import JudgeResponses, solve_minimax_weights_lp


def _probs(rng, N, C, n):
    return rng.dirichlet(np.ones(3), size=(N, C, n))


def _stochastic(rng, R, S, dep_scale=0.6):
    A = rng.random((R, R)) * dep_scale
    np.fill_diagonal(A, 0.0)
    B = rng.random((R, S))
    tot = A.sum(1, keepdims=True) + B.sum(1, keepdims=True)
    return A / tot, B / tot


def test_proposition1_duplicate_cycle_has_unit_spectral_radius():
    # p1 = p2, each witness uses only the other: A = [[0,1],[1,0]], B = 0.
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    B = np.zeros((2, 1))
    out = jr.compose(A, B)
    assert out["status"] == "spectral_radius_at_or_near_one"
    assert abs(out["rho"] - 1.0) < 1e-12


def test_composed_map_is_stochastic_on_survivors_and_satisfies_identity_and_bound():
    rng = np.random.default_rng(0)
    N, C, n = 8, 3, 50
    P = _probs(rng, N, C, n)
    rem, surv = [0, 1, 2], [3, 4, 5, 6, 7]
    A, B = _stochastic(rng, 3, 5)
    W = np.zeros((N, N))
    W[np.ix_(rem, rem)] = A
    W[np.ix_(rem, surv)] = B
    comp = jr.compose(A, B)
    assert comp["status"] == "ok"
    assert comp["M"].min() >= -1e-12
    assert np.abs(comp["M"].sum(1) - 1).max() < 1e-12
    assert np.all(comp["H"] >= np.eye(3) - 1e-12)            # H >= I
    d = jr.composition_on(P, rem, surv, W, comp)
    assert d["identity_max_abs"] < 1e-12
    assert np.all(d["row_errors"] <= d["row_bounds_context"] + 1e-12)
    assert np.all(d["row_bounds_context"] <= d["row_bounds"] + 1e-12)
    assert np.all(d["row_bounds"] >= d["epsilon"] - 1e-12)


def test_exact_witnesses_with_exit_compose_exactly():
    # p0 = 0.5 p1 + 0.5 p2, p1 = 0.5 p0 + 0.5 p2 with p2 retained -> M is exact.
    rng = np.random.default_rng(1)
    C, n = 2, 30
    p2 = rng.dirichlet(np.ones(3), size=(C, n))
    P = np.stack([p2, p2, p2])  # all equal: exact witnesses trivially
    A = np.array([[0.0, 0.5], [0.5, 0.0]])
    B = np.array([[0.5], [0.5]])
    W = np.zeros((3, 3))
    W[:2, :2] = A
    W[:2, 2:] = B
    comp = jr.compose(A, B)
    d = jr.composition_on(P, [0, 1], [2], W, comp)
    assert d["coverage_error"] < 1e-12 and d["bound"] < 1e-12
    assert np.allclose(comp["M"], 1.0)


def test_bootstrap_is_paired_and_pair_weighted():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(40), rng.integers(1, 4, size=40))
    n = len(groups)
    x = rng.random((2, 3, n))
    draws = jr.bootstrap_means(np.stack([x, x]), groups, 200, 7)
    assert np.array_equal(draws[:, 0], draws[:, 1])         # same multiplicities per arm
    # reproduce one draw by hand
    rng2 = np.random.default_rng(7)
    G = 40
    mult = rng2.multinomial(G, np.full(G, 1.0 / G), size=200)[0]
    w = mult[groups]
    manual = (x * w).sum(-1) / w.sum()
    assert np.allclose(draws[0, 0], manual)


def test_paired_comparison_identical_arms_gives_zero_delta():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(30), 2)
    L = rng.random((2, 4, 60))
    rows = jr.paired_comparison(L, L.copy(), groups, 0.2, 300, 11)
    for r in rows:
        assert r["delta"] == 0.0 and r["delta_ci_low"] == 0.0 and r["delta_ci_high"] == 0.0
        assert r["band_delta_low"] <= 0.0 <= r["band_delta_high"]


def test_bracket_contains_lp_optimum_mass_and_respects_slack():
    rng = np.random.default_rng(4)
    N, C, n = 6, 2, 40
    P = _probs(rng, N, C, n)
    resp = JudgeResponses([P[:, c].transpose(1, 0, 2) for c in range(C)])
    target, donors = 0, [1, 2, 3, 4, 5]
    opt, w = solve_minimax_weights_lp(resp, target, donors)
    br = jr.bracket_mass(P, target, donors, [1, 2], opt, 1e-6)
    mass = w[0] + w[1]
    assert br["min"]["mass"] <= mass + 1e-6 <= br["max"]["mass"] + 2e-6
    assert br["min"]["fit_error"] <= opt + 1e-5 and br["max"]["fit_error"] <= opt + 1e-5


def test_coordinate_deltas_band_contains_point():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(50), 2)
    a, b = rng.random((3, 4, 100)), rng.random((3, 4, 100))
    cd = jr.coordinate_deltas(a, b, groups, 400, 13)
    assert np.all(cd["band_low"] <= cd["point"]) and np.all(cd["point"] <= cd["band_high"])
    assert cd["maxt_q"] > 1.0


def test_removal_sets_order():
    assert jr.removal_sets(["a", "b", "c"]) == [
        ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]

