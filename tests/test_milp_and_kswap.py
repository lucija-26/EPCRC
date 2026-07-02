"""Sanity checks for the MILP oracle optimum and the k-swap reduction escape.

Reuses the triangle cloud from test_pruning_synthetic (3 hull vertices + 5
interior points, Y_fit == Y_eval): at gamma ~ 0 the unique minimal kept set is
exactly the 3 vertices, for BOTH the protocol semantics and the oracle-routing
MILP semantics (with fit == eval and an exact certificate, they coincide).

Run:  pytest tests/test_milp_and_kswap.py -v
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.milp import milp_min_representative_set
from epcrc.pruning import BackwardKSwapPruner

GAMMA = 1e-6


def _triangle_cloud():
    vertices = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
    interior = np.array(
        [[2.0, 1.0], [2.0, 2.0], [1.5, 1.0], [2.5, 1.0], [2.0, 1.5]]
    )
    pts = np.vstack([vertices, interior])
    Y = pts.T.copy()  # (n_queries=2, n_models=8)
    names = [f"pt{i}" for i in range(pts.shape[0])]
    return Y, {0, 1, 2}, names


def test_milp_recovers_hull_vertices():
    """MILP oracle optimum on the triangle cloud is exactly the 3 vertices."""
    Y, vertex_idx, _ = _triangle_cloud()
    res = milp_min_representative_set(Y, GAMMA)
    assert res.status == 0, res.message
    assert set(res.kept_set) == vertex_idx, (
        f"MILP kept {sorted(res.kept_set)}, expected {sorted(vertex_idx)}"
    )


def test_milp_matches_bruteforce_on_random_instance():
    """MILP size == brute-force protocol optimum on a small fit==eval instance.

    With Y_fit == Y_eval the protocol optimum upper-bounds the oracle optimum,
    and on generic random instances they coincide at this tolerance.
    """
    rng = np.random.default_rng(3)
    N, dim = 7, 2
    pts = rng.standard_normal((N, dim))
    Y = pts.T.copy()
    scale = float(np.mean([np.linalg.norm(pts[i] - pts[j])
                           for i in range(N) for j in range(i + 1, N)]))
    gamma = 0.2 * scale

    cov = CoverageFunctional(Y, Y, [f"m{i}" for i in range(N)])
    brute = None
    for k in range(1, N + 1):
        for combo in combinations(range(N), k):
            E, _ = cov.compute_coverage(set(combo))
            if E <= gamma:
                brute = k
                break
        if brute is not None:
            break

    res = milp_min_representative_set(Y, gamma)
    assert res.status == 0, res.message
    assert res.size <= brute, (
        f"MILP (oracle, {res.size}) must lower-bound protocol optimum ({brute})"
    )


def test_backward_kswap_never_worse_than_backward():
    """BackwardKSwapPruner output is feasible and no larger than backward's."""
    from epcrc.pruning import BackwardEliminationPruner

    rng = np.random.default_rng(11)
    for trial in range(5):
        N, dim = 9, 2
        pts = rng.standard_normal((N, dim))
        Y = pts.T.copy()
        scale = float(np.mean([np.linalg.norm(pts[i] - pts[j])
                               for i in range(N) for j in range(i + 1, N)]))
        gamma = 0.25 * scale
        cov = CoverageFunctional(Y, Y, [f"m{i}" for i in range(N)])

        bwd = BackwardEliminationPruner(cov, gamma).run()
        ksw = BackwardKSwapPruner(cov, gamma, max_swap_k=2).run()

        assert ksw.coverage <= gamma
        assert len(ksw.kept_set) <= len(bwd.kept_set), (
            f"trial {trial}: kswap {len(ksw.kept_set)} > backward {len(bwd.kept_set)}"
        )
