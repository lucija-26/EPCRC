"""Three-class judge outputs: total-variation loss and worst-context fitting.

An LLM judge asked to compare two responses returns a distribution over three
outcomes (A better, B better, tie), not a single number, and it is measured
under several contexts (clean items, position swaps, verbosity perturbations,
...).  This module carries those two changes into the coverage layer:

  * the per-item loss is total variation, ell(p, phat) = 0.5 * ||p - phat||_1;
  * one weight vector per target judge must work in *every* context, and the
    fit is scored by the worst one.

Because the maximum over contexts cannot be minimised directly, the fit is
written with an epigraph variable u that upper-bounds every context's mean loss.
`solve_minimax_weights_lp` states that as one linear program; `solve_minimax_weights`
solves the same problem by cutting planes, which avoids the per-item slack
variables and is what the experiments actually call.

`JudgeCoverageFunctional` exposes exactly the interface `CoverageFunctional`
does, so `pruning`, `milp`, `risk` and `task_error` run against judge panels
without modification.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .coverage import CoverageFunctional, SubstitutionCertificate

# Probability rows are allowed to drift this far from a valid distribution
# before we refuse the input; logprob-derived probabilities are renormalised
# upstream and land well inside this.
_SIMPLEX_TOL = 1e-4


class JudgeResponses:
    """Three-class judge probabilities grouped by evaluation context.

    ``blocks[c]`` has shape ``(n_items_c, n_judges, 3)`` and every row
    ``blocks[c][x, j]`` is a distribution over (A better, B better, tie).
    Contexts may hold different numbers of items; each context's loss is
    averaged over its own items before the maximum is taken, so a large clean
    split cannot drown out a small stress split.
    """

    def __init__(
        self,
        blocks: Sequence[np.ndarray],
        context_names: Optional[Sequence[str]] = None,
    ):
        self.blocks = [np.asarray(b, dtype=float) for b in blocks]

        if not self.blocks:
            raise ValueError("JudgeResponses needs at least one context block")

        for c, block in enumerate(self.blocks):
            if block.ndim != 3 or block.shape[2] != 3:
                raise ValueError(
                    f"context {c}: expected shape (n_items, n_judges, 3), "
                    f"got {block.shape}"
                )
            if block.shape[0] == 0 or block.shape[1] == 0:
                raise ValueError(f"context {c} is empty")
            if block.min() < -_SIMPLEX_TOL:
                raise ValueError(f"context {c} has negative probabilities")
            drift = float(np.abs(block.sum(axis=2) - 1.0).max())
            if drift > _SIMPLEX_TOL:
                raise ValueError(
                    f"context {c}: probability rows deviate from 1 by {drift:.2e}"
                )

        widths = {block.shape[1] for block in self.blocks}
        if len(widths) != 1:
            raise ValueError(f"contexts disagree on the number of judges: {widths}")

        self.n_judges = widths.pop()
        self.context_names = (
            list(context_names)
            if context_names is not None
            else [f"context_{c}" for c in range(len(self.blocks))]
        )
        if len(self.context_names) != len(self.blocks):
            raise ValueError("context_names length does not match the number of blocks")

    @property
    def n_contexts(self) -> int:
        return len(self.blocks)

    @property
    def n_items(self) -> int:
        """Total number of items across all contexts."""
        return int(sum(block.shape[0] for block in self.blocks))

    def subset(self, judge_indices: Sequence[int]) -> "JudgeResponses":
        idx = list(judge_indices)
        return JudgeResponses([b[:, idx, :] for b in self.blocks], self.context_names)


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation between three-class rows, along the last axis."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


def context_errors(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
    weights: np.ndarray,
) -> np.ndarray:
    """Mean TV error of the reconstruction of `target_idx`, one value per context."""
    kept_list = list(kept_list)
    w = np.asarray(weights, dtype=float).reshape(-1)

    out = np.empty(responses.n_contexts, dtype=float)
    for c, block in enumerate(responses.blocks):
        recon = np.tensordot(block[:, kept_list, :], w, axes=([1], [0]))
        out[c] = float(total_variation(block[:, target_idx, :], recon).mean())
    return out


def worst_context_error(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
    weights: np.ndarray,
) -> float:
    return float(context_errors(responses, target_idx, kept_list, weights).max())


def solve_minimax_weights_lp(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
) -> Tuple[float, np.ndarray]:
    """Reference implementation of the minimax fit, as one monolithic LP.

    Solves, over w in the simplex on `kept_list`,

        min_w max_c  (1 / |Q_c|) sum_{x in Q_c} TV(p_target(x), sum_j w_j p_j(x)).

    The epigraph reformulation introduces u (the ceiling on every context) and
    one slack t per item.  Both p and the reconstruction are distributions on
    three outcomes, so TV equals max_k |p_k - phat_k| and `t >= |residual_k|`
    for k = 0, 1, 2 pins t to exactly the item's TV at the optimum.

    Exact, but it scales badly.  The per-item slacks make the program
    ``m + n_items + 1`` wide and ``6 * n_items + n_contexts`` tall, so at panel
    scale one solve costs tens of seconds even though the object we actually
    want is only ``m``-dimensional.  `solve_minimax_weights` therefore defaults
    to the cutting-plane method, and this function stays as the ground truth
    that method is tested against.

    Returns (worst-context fit error, weights).
    """
    kept_list = list(kept_list)
    m = len(kept_list)

    if m == 0:
        return float("inf"), np.array([], dtype=float)

    if m == 1:
        w = np.array([1.0])
        return worst_context_error(responses, target_idx, kept_list, w), w

    n_items = responses.n_items
    n_var = m + n_items + 1  # weights, per-item slack, epigraph ceiling
    u_col = n_var - 1

    # --- item constraints: t_x >= |residual_k| for each class k ---------------
    # Row "plus"  encodes  -t_x + sum_j w_j p_j[k] <= p_target[k]
    # Row "minus" encodes  -t_x - sum_j w_j p_j[k] <= -p_target[k]
    w_coef: List[np.ndarray] = []
    t_index: List[np.ndarray] = []
    b_item: List[np.ndarray] = []

    # --- context constraints: mean_x t_x - u <= 0 -----------------------------
    ctx_rows: List[np.ndarray] = []
    ctx_cols: List[np.ndarray] = []
    ctx_vals: List[np.ndarray] = []

    offset = 0
    for c, block in enumerate(responses.blocks):
        n_c = block.shape[0]
        peers = block[:, kept_list, :]        # (n_c, m, 3)
        target = block[:, target_idx, :]      # (n_c, 3)

        # Flatten to one row per (item, class) pair.
        peer_rows = peers.transpose(0, 2, 1).reshape(n_c * 3, m)
        target_rows = target.reshape(n_c * 3)
        items = np.repeat(np.arange(offset, offset + n_c), 3)

        w_coef.append(peer_rows)
        w_coef.append(-peer_rows)
        t_index.append(items)
        t_index.append(items)
        b_item.append(target_rows)
        b_item.append(-target_rows)

        t_cols = m + np.arange(offset, offset + n_c)
        ctx_rows.append(np.full(n_c + 1, c))
        ctx_cols.append(np.concatenate([t_cols, [u_col]]))
        ctx_vals.append(np.concatenate([np.full(n_c, 1.0 / n_c), [-1.0]]))

        offset += n_c

    W = np.vstack(w_coef)                       # (6 * n_items, m)
    t_idx = np.concatenate(t_index)             # (6 * n_items,)
    b_ub = np.concatenate(b_item + [np.zeros(responses.n_contexts)])

    n_item_rows = W.shape[0]
    item_row_ids = np.arange(n_item_rows)

    rows = np.concatenate([
        np.repeat(item_row_ids, m),                      # weight block
        item_row_ids,                                    # slack block
        n_item_rows + np.concatenate(ctx_rows),          # context block
    ])
    cols = np.concatenate([
        np.tile(np.arange(m), n_item_rows),
        m + t_idx,
        np.concatenate(ctx_cols),
    ])
    vals = np.concatenate([
        W.reshape(-1),
        np.full(n_item_rows, -1.0),
        np.concatenate(ctx_vals),
    ])

    A_ub = coo_matrix(
        (vals, (rows, cols)),
        shape=(n_item_rows + responses.n_contexts, n_var),
    ).tocsr()

    A_eq = coo_matrix(
        (np.ones(m), (np.zeros(m, dtype=int), np.arange(m))), shape=(1, n_var)
    ).tocsr()

    objective = np.zeros(n_var)
    objective[u_col] = 1.0

    result = linprog(
        objective,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n_var,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            f"minimax fit failed for target {target_idx} on {m} judges: {result.message}"
        )

    w = np.clip(result.x[:m], 0.0, None)
    total = float(w.sum())
    w = w / total if total > 0 else np.full(m, 1.0 / m)

    # Report the error the returned (renormalised) weights actually achieve
    # rather than the LP's u, so the certificate is self-consistent.
    return worst_context_error(responses, target_idx, kept_list, w), w


def _objective_and_subgradients(
    peers: List[np.ndarray],
    targets: List[np.ndarray],
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Each context's mean TV at `w`, and a subgradient of each in w.

    ``f_c(w) = (1 / n_c) sum_x 0.5 sum_k |r_xk|`` with ``r = P_c w - q_c`` is
    convex and piecewise linear, so differentiating through the absolute value
    gives the subgradient ``(1 / n_c) sum_x 0.5 sum_k sign(r_xk) p_j(x)[k]``.
    Items where a residual is exactly zero contribute ``sign(0) = 0``, which is
    a valid element of the subdifferential.
    """
    vals = np.empty(len(peers), dtype=float)
    grads = np.empty((len(peers), w.size), dtype=float)
    for c, (peer, target) in enumerate(zip(peers, targets)):
        residual = np.tensordot(peer, w, axes=([1], [0])) - target
        vals[c] = 0.5 * np.abs(residual).sum(axis=1).mean()
        grads[c] = (
            0.5 * np.einsum("xk,xjk->j", np.sign(residual), peer) / residual.shape[0]
        )
    return vals, grads


# Weight on the mean-error tie-breaker.  It has to sit in a window: large enough
# that the F-difference it induces between tied weight vectors (order
# _TIE_BREAK * 0.1) clears the convergence tolerance, and small enough that it
# cannot move a reported figure.  At 1e-5 against tol 1e-7 the margin is 10x, and
# the worst-context error is held within 1e-7 of the unregularised optimum on the
# real panel -- three orders of magnitude below the four decimals that are
# reported.  Dropping it to 1e-6 is already too weak to break the tie.
_TIE_BREAK = 1e-5

# A residual gap no larger than this is accepted after `max_iter` instead of
# raising.  The master program is solved by HiGHS at its default feasibility
# tolerance of about 1e-7, so `lower` is itself only accurate to roughly `tol`,
# and the loop can then spend every iteration chasing a gap that has already
# stalled at that floor.  That is not hypothetical: it aborted a Core-20 C3 run
# at the fourth split seed, three hours in, with "gap 1.000e-07 > tol 1.000e-07"
# -- a gap and a tolerance that print identically.
#
# Accepting is safe here in a way it would not be for a certified bound.  The
# value returned is `worst_context_error` evaluated at the best iterate, so it is
# an error the returned weights actually achieve, not an LP estimate; the gap
# only bounds how far that achievable error sits above the optimal one.  At 1e-5
# that slack is the same order as `_TIE_BREAK`, which the fit already admits, and
# two orders below the four decimals anything reported carries.  A genuinely
# unconverged fit has a gap orders of magnitude larger and still raises.
_STALL_TOL = 1e-5


def solve_minimax_weights_cuts(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
    tol: float = 1e-7,
    max_iter: int = 500,
) -> Tuple[float, np.ndarray]:
    """The same minimax fit by cutting planes, without the per-item slacks.

    The monolithic LP in `solve_minimax_weights_lp` pays for one slack variable
    per item in order to express a maximum over items that it never actually
    needs to see.  Here that inner maximum is evaluated in closed form instead,
    so the master program is `m + 2` wide rather than `m + n_items + 1`, which is
    where the speedup comes from.  The objective minimised is

        F(w) = max_c f_c(w) + _TIE_BREAK * mean_c f_c(w),

    and the master keeps, over the iterates `w_i` visited so far,

        min  u + _TIE_BREAK * v
        s.t. u >= f_c(w_i) + g_ci . (w - w_i)      for every context c
             v >= mean_c f_c(w_i) + gbar_i . (w - w_i)
             w in the simplex.

    Every cut is a supporting hyperplane of a convex function, so the master's
    optimum bounds `F` from below while the best `F` actually evaluated bounds it
    from above, and the loop exits once that interval closes.  If `max_iter` runs
    out with the interval still open, a gap up to `_STALL_TOL` is accepted with a
    warning and anything larger raises; see that constant for why the distinction
    is needed and why accepting is sound.

    The tie-breaking term is not cosmetic.  The worst-context objective alone is
    often flat over an entire face of the simplex: as soon as one context is
    unreachably bad it pins the maximum by itself, and the remaining contexts can
    then vary freely without changing the objective at all.  A solver handed that
    problem returns an arbitrary point of the face, which would leave the
    per-context errors reported next to the fit an artefact of pivoting rather
    than a property of the panel.  Preferring the tied weights with the lowest
    mean error makes the fit well defined, at the cost of admitting a
    worst-context error up to `_TIE_BREAK` above the true optimum -- two orders
    of magnitude below anything that gets reported.

    Returns (worst-context fit error, weights), matching `solve_minimax_weights_lp`.
    """
    kept_list = list(kept_list)
    m = len(kept_list)

    if m == 0:
        return float("inf"), np.array([], dtype=float)

    if m == 1:
        w = np.array([1.0])
        return worst_context_error(responses, target_idx, kept_list, w), w

    peers = [block[:, kept_list, :] for block in responses.blocks]
    targets = [block[:, target_idx, :] for block in responses.blocks]
    n_contexts = len(peers)

    # Variables are [w (m), u, v]; u carries the max and v the mean.
    objective = np.concatenate([np.zeros(m), [1.0, _TIE_BREAK]])
    simplex_row = np.concatenate([np.ones(m), [0.0, 0.0]])[None, :]
    bounds = [(0.0, 1.0)] * m + [(None, None)] * 2

    max_block = np.hstack([-np.ones((n_contexts, 1)), np.zeros((n_contexts, 1))])
    mean_block = np.array([[0.0, -1.0]])

    cut_A = np.empty((0, m + 2), dtype=float)
    cut_b = np.empty(0, dtype=float)

    w = np.full(m, 1.0 / m)
    best_value, best_w = np.inf, w.copy()
    lower = -np.inf

    for _ in range(max_iter):
        vals, grads = _objective_and_subgradients(peers, targets, w)
        value = float(vals.max()) + _TIE_BREAK * float(vals.mean())
        if value < best_value:
            best_value, best_w = value, w.copy()

        mean_grad = grads.mean(axis=0, keepdims=True)
        cut_A = np.vstack([
            cut_A,
            np.hstack([grads, max_block]),
            np.hstack([mean_grad, mean_block]),
        ])
        cut_b = np.concatenate([
            cut_b,
            grads @ w - vals,
            mean_grad @ w - vals.mean(),
        ])

        if best_value - lower <= tol:
            break

        result = linprog(
            objective,
            A_ub=cut_A, b_ub=cut_b,
            A_eq=simplex_row, b_eq=np.array([1.0]),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(
                f"cutting-plane master LP failed for target {target_idx} "
                f"on {m} judges: {result.message}"
            )
        w = result.x[:m]
        lower = float(result.x[m] + _TIE_BREAK * result.x[m + 1])
    else:
        gap = best_value - lower
        if gap > _STALL_TOL:
            raise RuntimeError(
                f"cutting-plane fit for target {target_idx} on {m} judges did not "
                f"converge in {max_iter} iterations "
                f"(gap {gap:.3e} > tol {tol:.3e})"
            )
        warnings.warn(
            f"cutting-plane fit for target {target_idx} on {m} judges stalled at "
            f"gap {gap:.3e} after {max_iter} iterations, short of tol {tol:.3e}. "
            f"Accepting the best iterate: the error returned is one the weights "
            f"achieve, and it is within {gap:.3e} of optimal.",
            RuntimeWarning,
            stacklevel=2,
        )

    w = np.clip(best_w, 0.0, None)
    total = float(w.sum())
    w = w / total if total > 0 else np.full(m, 1.0 / m)

    return worst_context_error(responses, target_idx, kept_list, w), w


def solve_minimax_weights(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
) -> Tuple[float, np.ndarray]:
    """Fit one simplex weight vector that minimises the worst context's mean TV.

    Delegates to the cutting-plane solver, which agrees with the monolithic LP
    in `solve_minimax_weights_lp` to far beyond reporting precision and is fast
    enough to run the panel-scale experiments.
    """
    return solve_minimax_weights_cuts(responses, target_idx, kept_list)


class JudgeCoverageFunctional(CoverageFunctional):
    """Coverage functional for three-class judge panels under worst-context TV.

    Deliberately does not call `CoverageFunctional.__init__`: the base class
    validates a 2D scalar matrix that has no judge-panel analogue.  Everything
    downstream (`compute_coverage`, `compute_sum_uniqueness`, `find_bottleneck`)
    is inherited unchanged, because it only ever calls `compute_certificate`.
    """

    def __init__(
        self,
        fit: JudgeResponses,
        eval: JudgeResponses,
        model_names: Optional[List[str]] = None,
    ):
        if fit.n_judges != eval.n_judges:
            raise ValueError(
                f"judge dimension mismatch: fit has {fit.n_judges}, eval has {eval.n_judges}"
            )

        self.fit = fit
        self.eval = eval
        self.N = fit.n_judges
        self.metric = "tv"
        self.model_names = model_names or [f"judge_{i}" for i in range(self.N)]
        self._cache = {}

    def compute_certificate(
        self, target_idx: int, kept_set: Set[int]
    ) -> SubstitutionCertificate:
        kept_set = set(kept_set)
        kept_list = sorted(kept_set)

        if target_idx in kept_set:
            w = np.zeros(len(kept_list), dtype=float)
            w[kept_list.index(target_idx)] = 1.0
            return SubstitutionCertificate(
                model_idx=target_idx,
                model_name=self.model_names[target_idx],
                kept_set=kept_set,
                weights=w,
                uniqueness=0.0,
            )

        if not kept_list:
            return SubstitutionCertificate(
                model_idx=target_idx,
                model_name=self.model_names[target_idx],
                kept_set=kept_set,
                weights=np.array([], dtype=float),
                uniqueness=float("inf"),
            )

        _, w = solve_minimax_weights(self.fit, target_idx, kept_list)
        u = worst_context_error(self.eval, target_idx, kept_list, w)

        return SubstitutionCertificate(
            model_idx=target_idx,
            model_name=self.model_names[target_idx],
            kept_set=kept_set,
            weights=w,
            uniqueness=u,
        )
