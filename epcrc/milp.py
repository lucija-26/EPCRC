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
    best_bound: float = float("nan")  # best proven dual bound on |S| (lower bound)
    solver: str = "highs"


def milp_min_representative_set(
    Y_eval: np.ndarray,
    gamma: float,
    metric: str = "mean_abs",
    time_limit: Optional[float] = None,
    mip_rel_gap: float = 0.0,
    max_support: Optional[int] = None,
    solver: str = "highs",
    threads: Optional[int] = None,
    initial_set: Optional[List[int]] = None,
    mip_focus: Optional[int] = None,
    min_size: Optional[int] = None,
) -> MilpResult:
    """Exact oracle-routing minimum representative set via MILP.

    Variable layout (x has length N + N*N + n*N [+ N*N]):
      x[0:N]                    z_j     binary keep indicators
      x[N + i*N + j]            w_ij    routing weight of target i on model j
      x[N + N*N + i*n + t]      e_ti    |residual| envelope for target i, row t
      x[... + i*N + j]          u_ij    binary support indicators (sparse mode)

    max_support = r enforces ||w_i||_0 <= r for every certificate (the sparse
    substitution variant, paper Open Problem 5 / eq. 11): binary u_ij with
    w_ij <= u_ij and sum_j u_ij <= r.

    solver = "highs" (scipy, single-threaded) or "gurobi" (parallel branch-and-
    bound).  Both consume the identical constraint matrices built below, so the
    formulation is provably the same; only the search differs.  HiGHS does not
    converge on the full N=20 ecosystem within minutes -- use Gurobi there.

    initial_set supplies a known-feasible kept set (e.g. the backward/k-swap
    result) as a MIP start.  The LP relaxation of this formulation is weak --
    fractional z spreads weight across many models, so the dual bound stalls
    near 2 -- which means branch-and-bound burns its whole budget rediscovering
    an incumbent the greedy already has.  Seeding the incumbent lets the solver
    spend its budget proving the lower bound instead.  Gurobi only.

    min_size adds the cardinality cut sum_j z_j >= min_size.  Use it with a
    bound proven by `certify_lower_bound`; that is what actually closes the gap,
    since no cut on w can help (the e-envelope is already LP-tight, so the
    relaxation's weakness lives entirely in fractional z).  Gurobi only.

    mip_focus maps to Gurobi's MIPFocus (3 = focus on the bound).
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

    if solver == "gurobi":
        return _solve_gurobi(
            A, lb_rows, ub_rows, lb, ub, integrality, c,
            N, n_z, off_w, off_e, time_limit, mip_rel_gap, threads,
            initial_set, mip_focus, min_size,
        )
    if solver != "highs":
        raise ValueError(f"solver must be 'highs' or 'gurobi'; got {solver!r}")

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
    bound = float(getattr(res, "mip_dual_bound", float("nan")))
    return MilpResult(kept, len(kept), weights, int(res.status), str(res.message), gap,
                      dt, bound, "highs")


def _solve_gurobi(
    A, lb_rows, ub_rows, lb, ub, integrality, c,
    N, n_z, off_w, off_e, time_limit, mip_rel_gap, threads,
    initial_set=None, mip_focus=None, min_size=None,
) -> MilpResult:
    """Solve the already-built matrices with Gurobi's matrix API.

    Two-sided rows are added as A x >= lb_rows and A x <= ub_rows, skipping the
    infinite sides so no spurious constraints are introduced.
    """
    import gurobipy as gp
    from gurobipy import GRB

    A = A.tocsr()
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
        if time_limit is not None:
            m.Params.TimeLimit = float(time_limit)
        m.Params.MIPGap = float(mip_rel_gap)
        if threads is not None:
            m.Params.Threads = int(threads)
        if mip_focus is not None:
            m.Params.MIPFocus = int(mip_focus)

        vtype = np.where(integrality > 0.5, GRB.BINARY, GRB.CONTINUOUS)
        x = m.addMVar(len(c), lb=lb, ub=ub, vtype=vtype)

        fin_ub = np.isfinite(ub_rows)
        if fin_ub.any():
            m.addConstr(A[fin_ub] @ x <= ub_rows[fin_ub])
        fin_lb = np.isfinite(lb_rows)
        if fin_lb.any():
            m.addConstr(A[fin_lb] @ x >= lb_rows[fin_lb])

        if min_size is not None:
            m.addConstr(x[:n_z].sum() >= float(min_size))

        if initial_set is not None:
            seed = np.zeros(n_z)
            seed[list(initial_set)] = 1.0
            x[:n_z].Start = seed

        m.setObjective(c @ x, GRB.MINIMIZE)

        t0 = time.time()
        m.optimize()
        dt = time.time() - t0

        bound = float(m.ObjBound) if m.SolCount > 0 or m.Status == GRB.OPTIMAL else float("nan")
        if m.SolCount == 0:
            return MilpResult([], -1, np.zeros((N, N)), int(m.Status),
                              f"gurobi status {m.Status}, no incumbent",
                              float("nan"), dt, bound, "gurobi")

        xv = x.X
        z = xv[:n_z]
        kept = [j for j in range(N) if z[j] > 0.5]
        weights = xv[off_w:off_e].reshape(N, N)
        # scipy convention: status 0 == optimal
        status = 0 if m.Status == GRB.OPTIMAL else int(m.Status)
        return MilpResult(kept, len(kept), weights, status,
                          f"gurobi status {m.Status}", float(m.MIPGap), dt,
                          bound, "gurobi")


def min_substitution_error(
    Y_eval: np.ndarray,
    target: int,
    subset,
    metric: str = "mean_abs",
) -> float:
    """Best achievable substitution error for `target` routed onto `subset`.

    Solves min ||Y_S w - y_i|| over the simplex on S -- a tiny LP in |S| + n
    (mean_abs) or |S| + 1 (max) variables.  The oracle-routing feasibility of a
    keep set S is exactly `min_substitution_error(Y, i, S) <= gamma for all i`,
    and it decomposes over targets, which is what makes exhaustive
    certification of small subsets cheap.
    """
    from scipy.optimize import linprog

    Y = np.asarray(Y_eval, dtype=float)
    n, _ = Y.shape
    S = list(subset)
    k = len(S)
    if k == 0:
        return float("inf")
    Y_S = Y[:, S]
    y_i = Y[:, target]

    if metric == "mean_abs":
        n_e = n
        E = sparse.eye(n)
        c_e = np.full(n, 1.0 / n)
    elif metric == "max":
        n_e = 1
        E = np.ones((n, 1))
        c_e = np.ones(1)
    else:
        raise ValueError(f"metric must be 'mean_abs' or 'max'; got {metric!r}")

    A_ub = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(Y_S), -E]),
        sparse.hstack([sparse.csr_matrix(-Y_S), -E]),
    ], format="csc")
    b_ub = np.concatenate([y_i, -y_i])
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(k), np.zeros(n_e)])[None, :])

    res = linprog(
        c=np.concatenate([np.zeros(k), c_e]),
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * k + [(0.0, None)] * n_e,
        method="highs",
    )
    return float(res.fun) if res.success else float("inf")


def is_feasible_set(Y_eval, gamma, subset, metric="mean_abs", order=None, tol=1e-9) -> bool:
    """True iff every target can be gamma-covered by routing onto `subset`.

    `order` lets the caller test historically-hard targets first; the scan exits
    on the first target that fails, which is the bulk of the speedup during
    exhaustive certification.
    """
    N = np.asarray(Y_eval).shape[1]
    for i in (order if order is not None else range(N)):
        if min_substitution_error(Y_eval, i, subset, metric) > gamma + tol:
            return False
    return True


@dataclass
class CertificateResult:
    lower_bound: int          # proven: no feasible set of size < lower_bound exists
    best_set: Optional[List[int]]  # feasible set found during the scan, if any
    exhausted_upto: int       # every subset of size <= this was enumerated
    n_subsets_tested: int
    time_s: float


_CERT: dict = {}


def _cert_init(Y, gamma, metric):
    # Each worker keeps its own refutation counter; the ordering heuristic only
    # needs to be locally good, not globally consistent.
    _CERT.update(Y=Y, gamma=gamma, metric=metric,
                 fail_counts=np.zeros(Y.shape[1], dtype=int))


def _cert_check(subset):
    Y, gamma, metric = _CERT["Y"], _CERT["gamma"], _CERT["metric"]
    fc = _CERT["fail_counts"]
    for i in np.argsort(-fc):
        if min_substitution_error(Y, int(i), subset, metric) > gamma + 1e-9:
            fc[i] += 1
            return None
    return subset


def certify_lower_bound(
    Y_eval: np.ndarray,
    gamma: float,
    max_k: int,
    metric: str = "mean_abs",
    time_limit: Optional[float] = None,
    n_jobs: int = 1,
) -> CertificateResult:
    """Prove |OPT| > k by exhausting every keep set of size k, for k = 1..max_k.

    This is the piece the MILP cannot supply.  Branch-and-bound stalls because
    the LP relaxation lets fractional z mix models at half cost, so the dual
    bound sits near 2 no matter how long it runs.  Enumeration sidesteps the
    relaxation entirely: if no subset of size k is feasible then |OPT| >= k + 1
    is a *proof*, and feeding it back as the cardinality cut `min_size` closes
    the gap immediately.

    Stops early and returns the witness if some subset of size k is feasible --
    combined with a greedy incumbent of the same size, that settles optimality.
    """
    import multiprocessing as mp
    from itertools import combinations

    Y = np.asarray(Y_eval, dtype=float)
    N = Y.shape[1]
    t0 = time.time()
    tested = 0
    top_k = min(max_k, N)

    if n_jobs == 1:
        _cert_init(Y, gamma, metric)
        for k in range(1, top_k + 1):
            for subset in combinations(range(N), k):
                if time_limit is not None and time.time() - t0 > time_limit:
                    return CertificateResult(k, None, k - 1, tested, time.time() - t0)
                tested += 1
                if _cert_check(subset) is not None:
                    return CertificateResult(k, list(subset), k - 1, tested,
                                             time.time() - t0)
        return CertificateResult(top_k + 1, None, top_k, tested, time.time() - t0)

    with mp.Pool(n_jobs, initializer=_cert_init, initargs=(Y, gamma, metric)) as pool:
        for k in range(1, top_k + 1):
            subsets = list(combinations(range(N), k))
            for hit in pool.imap_unordered(_cert_check, subsets, chunksize=8):
                tested += 1
                if hit is not None:
                    pool.terminate()
                    return CertificateResult(k, list(hit), k - 1, tested,
                                             time.time() - t0)
            if time_limit is not None and time.time() - t0 > time_limit:
                return CertificateResult(k + 1, None, k, tested, time.time() - t0)
    return CertificateResult(top_k + 1, None, top_k, tested, time.time() - t0)
