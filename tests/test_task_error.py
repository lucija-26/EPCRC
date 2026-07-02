"""Sanity checks for the beta (task-error preservation) functional.

Triangle cloud again: with S = the 3 hull vertices, every model is either kept
or an EXACT convex combination of S, so every substitute reproduces its model
and delta(i|S) ~ 0 for any ground truth.

Run:  pytest tests/test_task_error.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.task_error import beta_coverage, joint_feasible, substitution_task_errors

GAMMA = 1e-6


def _triangle_cov():
    vertices = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
    interior = np.array(
        [[2.0, 1.0], [2.0, 2.0], [1.5, 1.0], [2.5, 1.0], [2.0, 1.5]]
    )
    pts = np.vstack([vertices, interior])
    Y = pts.T.copy()  # (2, 8)
    names = [f"pt{i}" for i in range(pts.shape[0])]
    return CoverageFunctional(Y, Y, names), {0, 1, 2}


def test_exact_substitution_has_zero_delta():
    cov, vertices = _triangle_cov()
    y_true = np.array([1.0, 2.0])  # arbitrary ground truth at the 2 queries
    errs = substitution_task_errors(cov, vertices, y_true, loss="rmse")
    for i, (err_orig, err_sub, delta) in errs.items():
        assert abs(delta) < 1e-5, f"model {i}: delta={delta} (expected ~0)"
    assert beta_coverage(cov, vertices, y_true) < 1e-5


def test_joint_feasibility_tightens_gamma_only():
    cov, vertices = _triangle_cov()
    y_true = np.array([1.0, 2.0])
    # gamma-feasible and beta-feasible with a loose beta:
    assert joint_feasible(cov, vertices, GAMMA, y_true, beta=1.0)
    # an infeasible-under-gamma set stays infeasible regardless of beta:
    assert not joint_feasible(cov, {0, 1}, GAMMA, y_true, beta=1.0)


def test_quality_prefilter_and_jensen_bound():
    """Bound 2: any simplex blend of eligible models stays within beta."""
    from epcrc.task_error import quality_eligible_set

    rng = np.random.default_rng(2)
    n, N = 300, 6
    y_true = rng.standard_normal(n)
    # Models with increasing noise -> increasing RMSE.
    Y = np.column_stack([y_true + (0.2 + 0.4 * j) * rng.standard_normal(n)
                         for j in range(N)])
    cov = CoverageFunctional(Y, Y, [f"m{j}" for j in range(N)])

    rmse = [float(np.sqrt(np.mean((Y[:, j] - y_true) ** 2))) for j in range(N)]
    beta = sorted(rmse)[2]  # third-best model's RMSE
    K = quality_eligible_set(cov, y_true, beta, loss="rmse")
    assert K == {j for j in range(N) if rmse[j] <= beta} and len(K) == 3

    # Jensen: every random simplex blend over K has RMSE <= beta.
    cols = sorted(K)
    for _ in range(50):
        w = rng.dirichlet(np.ones(len(cols)))
        blend_rmse = float(np.sqrt(np.mean((Y[:, cols] @ w - y_true) ** 2)))
        assert blend_rmse <= beta + 1e-9


def test_delta_can_be_negative():
    """A convex mixture can beat the original model on the task."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.standard_normal(n)
    # Two models = truth + opposite-sign noise; their average is closer to truth.
    noise = rng.standard_normal(n)
    Y = np.column_stack([y_true + noise, y_true - noise, y_true + 3.0 * noise])
    cov = CoverageFunctional(Y, Y, ["plus", "minus", "target"])
    errs = substitution_task_errors(cov, {0, 1}, y_true, loss="rmse")
    # The 'target' model (idx 2) is substituted by a mix of models 0 and 1;
    # any such mix has error <= its own 3-sigma noise -> delta < 0.
    assert errs[2][2] < 0, f"expected negative delta, got {errs[2][2]}"
