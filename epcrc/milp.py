"""MILP formulation of the minimal representative set problem (Open Problem 4).

Solves the *oracle-routing* variant of ecosystem pruning exactly:

    min  sum_j z_j
    s.t. for every model i:  there EXISTS w_i in the simplex, supported only on
         selected models (w_ij <= z_j), whose evaluation-sample substitution
         error is <= gamma.

This is a Mixed-Integer Linear Program because the mean-absolute (or max)
residual constraint linearises with auxiliary variables:

    e_ti >= +(y_ti - (Y w_i)_t)
    e_ti >= -(y_ti - (Y w_i)_t)
    mean_abs:  (1/n) * sum_t e_ti <= gamma        (one row per target i)
    max:       e_ti <= gamma                       (one row per (t, i))

IMPORTANT SEMANTICS: this optimum is a *lower bound* on the protocol-true
optimum used everywhere else in this repo.  The DISCO protocol fits w_i by
least squares on Y_fit and then scores it on Y_eval; the MILP instead picks
w_i directly to satisfy the evaluation-sample constraint (an oracle router).
Any protocol-feasible S is MILP-feasible (its fit weights witness the
existential), so

    |OPT_milp| <= |OPT_protocol| <= |greedy|.

The gap between the two optima measures how much the honest fit/eval split
costs relative to the best certificate that exists at all.

Solved with scipy.optimize.milp (HiGHS).  Swap in Gurobi on the server for
large N by translating `build_milp` -- the matrix layout is documented below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class MilpResult:
    kept_set: List[int]
    size: int
    weights: np.ndarray          # (N, N): row i = oracle routing weights for model i
    status: int                  # scipy milp status (0 = optimal)
    message: str
    mip_gap: float
    solve_time_s: float


def milp_min_representative_set(
    Y_eval: np.ndarray,
    gamma: float,
    metric: str = "mean_abs",
    time_limit: Optional[float] = None,
    mip_rel_gap: float = 0.0,
    max_support: Optional[int] = None,
) -> MilpResult:
    """Exact oracle-routing minimum representative set via MILP (HiGHS).

    Variable layout (x has length N + N*N + n*N [+ N*N]):
      x[0:N]                    z_j     binary keep indicators
      x[N + i*N + j]            w_ij    routing weight of target i on model j
      x[N + N*N + i*n + t]      e_ti    |residual| envelope for target i, row t
      x[... + i*N + j]          u_ij    binary support indicators (sparse mode)

    max_support = r enforces ||w_i||_0 <= r for every certificate (the sparse
    substitution variant, paper Open Problem 5 / eq. 11): binary u_ij with
    w_ij <= u_ij and sum_j u_ij <= r.
    """
    Y = np.asarray(Y_eval, dtype=float)
    n, N = Y.shape
    if metric not in ("mean_abs", "max"):
        raise ValueError(f"metric must be 'mean_abs' or 'max' (linear); got {metric!r}")

    n_z, n_w, n_e = N, N * N, N * n
    n_var = n_z + n_w + n_e
    off_w, off_e = n_z, n_z + n_w

    rows_A: List[sparse.spmatrix] = []
    lb_list: List[np.ndarray] = []
    ub_list: List[np.ndarray] = []

    # (1) Simplex: sum_j w_ij = 1 for each i.
    A_simplex = sparse.hstack([
        sparse.csr_matrix((N, n_z)),
        sparse.kron(sparse.eye(N), np.ones((1, N))),
        sparse.csr_matrix((N, n_e)),
    ])
    rows_A.append(A_simplex)
    lb_list.append(np.ones(N))
    ub_list.append(np.ones(N))

    # (2) Support: w_ij - z_j <= 0 for each (i, j).
    Z_block = sparse.vstack([-sparse.eye(N)] * N)             # (N*N, N)
    A_support = sparse.hstack([
        Z_block,
        sparse.eye(n_w),
        sparse.csr_matrix((n_w, n_e)),
    ])
    rows_A.append(A_support)
    lb_list.append(np.full(n_w, -np.inf))
    ub_list.append(np.zeros(n_w))

    # (3) Absolute-value envelope, per target i (n rows each, twice):
    #     (Y w_i)_t - e_ti <= y_ti      and     -(Y w_i)_t - e_ti <= -y_ti
    Y_sp = sparse.csr_matrix(Y)
    W_pos = sparse.block_diag([Y_sp] * N)                     # (N*n, N*N)
    E_eye = sparse.eye(n_e)
    A_abs_pos = sparse.hstack([sparse.csr_matrix((n_e, n_z)), W_pos, -E_eye])
    A_abs_neg = sparse.hstack([sparse.csr_matrix((n_e, n_z)), -W_pos, -E_eye])
    y_flat = Y.T.reshape(-1)                                   # y_ti in (i, t) order
    rows_A.extend([A_abs_pos, A_abs_neg])
    lb_list.extend([np.full(n_e, -np.inf), np.full(n_e, -np.inf)])
    ub_list.extend([y_flat, -y_flat])

    # (4) Error budget.
    if metric == "mean_abs":
        A_budget = sparse.hstack([
            sparse.csr_matrix((N, n_z + n_w)),
            sparse.kron(sparse.eye(N), np.full((1, n), 1.0 / n)),
        ])
        rows_A.append(A_budget)
        lb_list.append(np.full(N, -np.inf))
        ub_list.append(np.full(N, float(gamma)))
        e_ub = np.inf
    else:  # max: enforce e_ti <= gamma via variable bounds (no extra rows)
        e_ub = float(gamma)

    A = sparse.vstack(rows_A, format="csc")
    lb_rows = np.concatenate(lb_list)
    ub_rows = np.concatenate(ub_list)

    n_u = N * N if max_support is not None else 0
    if max_support is not None:
        # Pad existing rows with zero columns for u, then add:
        #   (5a) w_ij - u_ij <= 0            (support indicator coupling)
        #   (5b) sum_j u_ij <= max_support   (per-target sparsity budget)
        A = sparse.hstack([A, sparse.csr_matrix((A.shape[0], n_u))], format="csc")
        A_supp = sparse.hstack([
            sparse.csr_matrix((n_u, n_z)),
            sparse.eye(n_u),
            sparse.csr_matrix((n_u, n_e)),
            -sparse.eye(n_u),
        ])
        A_card = sparse.hstack([
            sparse.csr_matrix((N, n_z + n_w + n_e)),
            sparse.kron(sparse.eye(N), np.ones((1, N))),
        ])
        A = sparse.vstack([A, A_supp, A_card], format="csc")
        lb_rows = np.concatenate([lb_rows, np.full(n_u, -np.inf), np.full(N, -np.inf)])
        ub_rows = np.concatenate([ub_rows, np.zeros(n_u), np.full(N, float(max_support))])
        n_var += n_u

    constraint = LinearConstraint(A, lb_rows, ub_rows)

    lb = np.zeros(n_var)
    ub = np.concatenate([np.ones(n_z), np.ones(n_w), np.full(n_e, e_ub), np.ones(n_u)])
    integrality = np.concatenate([np.ones(n_z), np.zeros(n_w + n_e), np.ones(n_u)])

    c = np.concatenate([np.ones(n_z), np.zeros(n_w + n_e + n_u)])

    options = {"mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    t0 = time.time()
    res = milp(
        c=c,
        constraints=constraint,
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options=options,
    )
    dt = time.time() - t0

    if res.x is None:
        return MilpResult([], -1, np.zeros((N, N)), int(res.status), str(res.message),
                          float("nan"), dt)

    z = res.x[:n_z]
    kept = [j for j in range(N) if z[j] > 0.5]
    weights = res.x[off_w:off_e].reshape(N, N)
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return MilpResult(kept, len(kept), weights, int(res.status), str(res.message), gap, dt)
