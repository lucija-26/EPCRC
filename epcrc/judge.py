"""Three-class judge outputs: total-variation loss and worst-context fitting.

An LLM judge asked to compare two responses returns a distribution over three
outcomes (A better, B better, tie), not a single number, and it is measured
under several contexts (clean items, position swaps, verbosity perturbations,
...).  This module carries those two changes into the coverage layer:

  * the per-item loss is total variation, ell(p, phat) = 0.5 * ||p - phat||_1;
  * one weight vector per target judge must work in *every* context, and the
    fit is scored by the worst one.

Because the maximum over contexts cannot be minimised directly, the fit is
written as a linear program with an epigraph variable u that upper-bounds every
context's mean loss (see `solve_minimax_weights`).

`JudgeCoverageFunctional` exposes exactly the interface `CoverageFunctional`
does, so `pruning`, `milp`, `risk` and `task_error` run against judge panels
without modification.
"""

from __future__ import annotations

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


def solve_minimax_weights(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
) -> Tuple[float, np.ndarray]:
    """Fit one simplex weight vector that minimises the worst context's mean TV.

    Solves, over w in the simplex on `kept_list`,

        min_w max_c  (1 / |Q_c|) sum_{x in Q_c} TV(p_target(x), sum_j w_j p_j(x)).

    The epigraph reformulation introduces u (the ceiling on every context) and
    one slack t per item.  Both p and the reconstruction are distributions on
    three outcomes, so TV equals max_k |p_k - phat_k| and `t >= |residual_k|`
    for k = 0, 1, 2 pins t to exactly the item's TV at the optimum.

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
