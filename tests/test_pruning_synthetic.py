"""Task-1 sanity checks on a tiny synthetic cloud with a KNOWN answer.

Construction: 8 points in R^2 = 3 clear convex-hull vertices (a triangle) plus 5
points strictly inside the triangle.  Each model is a column of Y, so Y has shape
(n_queries=2, n_models=8) and Y_fit == Y_eval == the points (no honest split here,
we want the noiseless geometry).

Ground truth at gamma ~ 0:
  - The 3 triangle vertices are extreme points: no convex mix of the others can
    reproduce them, so U(vertex | others) is O(1) > 0.
  - Every interior point IS a convex combination of the vertices, so its
    simplex-projection residual is 0 (up to SLSQP tolerance).

Therefore:
  - BACKWARD must peel exactly the 5 interior points and return the 3 vertices,
    with no further feasible removal.
  - FORWARD must keep adding until E = 0, which is impossible unless all 3 vertices
    are in S; hence forward's output CONTAINS the 3 vertices (it may keep extras).

We use gamma = 1e-6 (not literally 0) so that ~1e-9 SLSQP residuals on the
interior points count as feasible while the O(1) vertex uniqueness does not.
Run:  pytest tests/test_pruning_synthetic.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import BackwardEliminationPruner, ForwardSelectionPruner

# gamma ~ 0, with a margin to absorb the SLSQP solver tolerance on interior points.
GAMMA = 1e-6


def _build_cloud():
    """3 triangle vertices + 5 strictly-interior points -> Y of shape (2, 8)."""
    vertices = np.array(
        [
            [0.0, 0.0],  # idx 0
            [4.0, 0.0],  # idx 1
            [2.0, 4.0],  # idx 2
        ]
    )
    interior = np.array(
        [
            [2.0, 1.0],  # idx 3
            [2.0, 2.0],  # idx 4
            [1.5, 1.0],  # idx 5
            [2.5, 1.0],  # idx 6
            [2.0, 1.5],  # idx 7
        ]
    )
    pts = np.vstack([vertices, interior])  # (8, 2)
    vertex_idx = {0, 1, 2}
    # Models are columns -> Y has shape (n_queries=2, n_models=8).
    Y = pts.T.copy()
    names = [f"pt{i}" for i in range(pts.shape[0])]
    return Y, vertex_idx, names


def _coverage():
    Y, vertex_idx, names = _build_cloud()
    cov = CoverageFunctional(Y_fit=Y, Y_eval=Y, model_names=names, metric="mean_abs")
    return cov, vertex_idx


def test_backward_returns_exact_hull_vertices():
    """Backward must return EXACTLY the 3 hull vertices, with E <= gamma and no
    further feasible removal."""
    cov, vertex_idx = _coverage()
    res = BackwardEliminationPruner(cov, GAMMA).run()

    assert res.kept_set == vertex_idx, (
        f"backward kept {sorted(res.kept_set)}, expected hull vertices "
        f"{sorted(vertex_idx)}"
    )
    # Returned set is feasible.
    assert res.coverage <= GAMMA, f"E(S)={res.coverage} > gamma={GAMMA}"

    # No single further removal stays feasible (it is a removal-local optimum).
    for j in sorted(res.kept_set):
        E_without, _ = cov.compute_coverage(res.kept_set - {j})
        assert E_without > GAMMA, (
            f"removing {j} kept E={E_without} <= gamma -> backward stopped early"
        )


def test_forward_output_contains_hull_vertices():
    """Forward's first feasible prefix must CONTAIN all 3 hull vertices, and be
    feasible (E <= gamma)."""
    cov, vertex_idx = _coverage()
    res = ForwardSelectionPruner(cov, GAMMA).run()

    assert vertex_idx.issubset(res.kept_set), (
        f"forward kept {sorted(res.kept_set)}, missing hull vertices "
        f"{sorted(vertex_idx - res.kept_set)}"
    )
    assert res.coverage <= GAMMA, f"E(S)={res.coverage} > gamma={GAMMA}"


def test_both_return_feasible_sets():
    """Every returned set must satisfy E(S) <= gamma."""
    cov, _ = _coverage()
    for pruner in (BackwardEliminationPruner(cov, GAMMA), ForwardSelectionPruner(cov, GAMMA)):
        res = pruner.run()
        assert res.coverage <= GAMMA, (
            f"{type(pruner).__name__} returned E(S)={res.coverage} > gamma={GAMMA}"
        )
