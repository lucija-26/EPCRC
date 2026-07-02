"""Sanity checks for sparse MILP certificates (OP5) and risk-controlled pruning (OP1).

Run:  pytest tests/test_sparse_and_risk.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.milp import milp_min_representative_set
from epcrc.pruning import BackwardEliminationPruner
from epcrc.risk import RiskControlledBackwardPruner, coverage_ucb, uniqueness_ucb

GAMMA = 1e-6


def _triangle_Y():
    vertices = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
    interior = np.array(
        [[2.0, 1.0], [2.0, 2.0], [1.5, 1.0], [2.5, 1.0], [2.0, 1.5]]
    )
    return np.vstack([vertices, interior]).T.copy()  # (2, 8)


def test_sparse_milp_r1_forces_keep_everything():
    """r=1 at gamma~0: no single model reproduces another exactly, so every
    model must self-route -> the optimum keeps all 8."""
    Y = _triangle_Y()
    res = milp_min_representative_set(Y, GAMMA, max_support=1)
    assert res.status == 0, res.message
    assert res.size == Y.shape[1]


def test_sparse_milp_r3_matches_unconstrained():
    """Certificates here mix at most 3 vertices, so r=3 costs nothing."""
    Y = _triangle_Y()
    dense = milp_min_representative_set(Y, GAMMA)
    sparse3 = milp_min_representative_set(Y, GAMMA, max_support=3)
    assert sparse3.status == 0, sparse3.message
    assert sparse3.size == dense.size == 3
    # support sizes actually respect the budget
    for i in range(Y.shape[1]):
        assert int((sparse3.weights[i] > 1e-6).sum()) <= 3


def _noisy_instance(seed=0, n=400, N=8, dim=2):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, dim))
    base = rng.standard_normal((n, dim))
    Y_fit = base @ pts.T + 0.05 * rng.standard_normal((n, N))
    Y_eval = base @ pts.T + 0.05 * rng.standard_normal((n, N))
    return CoverageFunctional(Y_fit, Y_eval, [f"m{i}" for i in range(N)])


def test_ucb_dominates_point_estimate():
    cov = _noisy_instance()
    S = {0, 1, 2}
    _, certs = cov.compute_coverage(S, return_certificates=True)
    ucb = uniqueness_ucb(cov, S, delta=0.05)
    for i in range(cov.N):
        assert ucb[i] >= certs[i].uniqueness - 1e-12


def test_risk_controlled_is_more_conservative():
    """UCB-gated backward keeps at least as many models as plain backward,
    and its returned set is point-estimate feasible."""
    cov = _noisy_instance()
    gamma = 0.5
    plain = BackwardEliminationPruner(cov, gamma).run()
    certified = RiskControlledBackwardPruner(cov, gamma, delta=0.05).run()
    assert len(certified.kept_set) >= len(plain.kept_set)
    assert certified.coverage <= gamma
    assert coverage_ucb(cov, certified.kept_set) <= gamma or \
        len(certified.kept_set) == cov.N
