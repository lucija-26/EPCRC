"""Individual-to-joint removal audit and the Theorem 2 composition arm.

Everything here works on dense arrays of shape ``[N, C, n, 3]`` (judge,
context, pair, label) for one partition.  Fitting is done elsewhere, by
`epcrc.judge.solve_minimax_weights`, on FIT only; this module scores frozen
weight rows, builds the composed map M = (I - A)^{-1} B from singleton rows,
evaluates the Theorem 2 bound, brackets the dependency mass over near-optimal
singleton witnesses, and runs the paired source-task bootstrap.

Loss order follows the paper: TV per pair, mean over pairs within a context,
then the maximum over contexts, then over removed targets.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from epcrc.judge import JudgeResponses


# --------------------------------------------------------------------------
# data layout
# --------------------------------------------------------------------------

def dense(responses: JudgeResponses) -> np.ndarray:
    """JudgeResponses (blocks of [n, N, 3]) -> one array [N, C, n, 3].

    Every context of the scored panel holds the same pairs in the same order,
    so the blocks stack along a common pair axis.
    """
    sizes = {b.shape[0] for b in responses.blocks}
    if len(sizes) != 1:
        raise ValueError(f"contexts hold different numbers of pairs: {sizes}")
    return np.stack([b.transpose(1, 0, 2) for b in responses.blocks], axis=1)


def removal_sets(targets: Sequence[str]) -> List[Tuple[str, ...]]:
    """All nonempty subsets, singletons first, in a fixed order."""
    return [r for k in range(1, len(targets) + 1) for r in combinations(targets, k)]


def stable_seed(base: int, *parts) -> int:
    text = "|".join(map(str, (base,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def reconstruct(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weight rows [T, N] over all judge columns -> predictions [T, C, n, 3]."""
    return np.einsum("ij,jcnd->icnd", W, P, optimize=True)


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=-1)


def row_losses(P: np.ndarray, targets: Sequence[int], W: np.ndarray) -> np.ndarray:
    """Per-pair TV of each target's reconstruction: [T, C, n]."""
    return tv(P[list(targets)], reconstruct(P, W))


def summarise(loss: np.ndarray) -> Dict[str, object]:
    """[T, C, n] -> context means, per-target worst context, coverage error."""
    means = loss.mean(axis=-1)
    t, c = np.unravel_index(int(np.argmax(means)), means.shape)
    return {
        "context_means": means,
        "endpoint_errors": means.max(axis=1),
        "coverage_error": float(means.max()),
        "worst_target": int(t),
        "worst_context": int(c),
    }


# --------------------------------------------------------------------------
# Theorem 2
# --------------------------------------------------------------------------

def split_witness(W_single: np.ndarray, removed: Sequence[int],
                  survivors: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of the singleton witnesses for `removed`, split into A (removed) and B (survivors)."""
    rows = W_single[list(removed)]
    return rows[:, list(removed)], rows[:, list(survivors)]


def compose(A: np.ndarray, B: np.ndarray, weight_tol: float = 1e-8,
            spectral_margin: float = 1e-10, condition_limit: float = 1e10) -> Dict[str, object]:
    """Build H = (I - A)^{-1} and M = H B, or return a specific failure status."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if min(A.min(), B.min()) < -weight_tol:
        raise ValueError("witness weights must be nonnegative")
    row_err = float(np.abs(A.sum(1) + B.sum(1) - 1.0).max())
    if row_err > weight_tol:
        raise ValueError(f"witness rows are not stochastic (error {row_err:.2e})")
    system = np.eye(len(A)) - A
    out: Dict[str, object] = {
        "rho": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "condition_number": float(np.linalg.cond(system)),
        "dependency_mass": A.sum(1),
        "survivor_mass": B.sum(1),
    }
    if out["rho"] >= 1.0 - spectral_margin:
        out["status"] = "spectral_radius_at_or_near_one"
        return out
    if not np.isfinite(out["condition_number"]) or out["condition_number"] > condition_limit:
        out["status"] = "ill_conditioned"
        return out
    H = np.linalg.solve(system, np.eye(len(A)))
    M = np.linalg.solve(system, B)
    out.update(
        H=H, M=M,
        min_M=float(M.min()),
        row_sum_error=float(np.abs(M.sum(1) - 1.0).max()),
        expected_depth=H.sum(1),
    )
    ok = M.min() >= -weight_tol and out["row_sum_error"] <= max(10 * weight_tol, 1e-7)
    out["status"] = "ok" if ok else "simplex_check_failed"
    return out


def composition_on(P: np.ndarray, removed: Sequence[int], survivors: Sequence[int],
                   W_single: np.ndarray, comp: Dict[str, object]) -> Dict[str, object]:
    """Score the composed map on one partition and evaluate the bound there.

    epsilon is the singleton residual measured on this same partition, so the
    bound is an empirical bound for this sample (paper eq. 8 with P replaced by
    the sample).
    """
    removed = list(removed)
    survivors = list(survivors)
    H, M = comp["H"], comp["M"]
    target = P[removed]
    single_pred = reconstruct(P, W_single[removed])
    D = target - single_pred                                  # [R, C, n, 3]
    single_loss = 0.5 * np.abs(D).sum(-1)                      # [R, C, n]
    eps_ctx = single_loss.mean(-1)                             # [R, C]
    eps = eps_ctx.max(1)                                       # [R]

    W_comp = np.zeros((len(removed), P.shape[0]))
    W_comp[:, survivors] = M
    comp_pred = reconstruct(P, W_comp)
    comp_loss = tv(target, comp_pred)                          # [R, C, n]
    rows = comp_loss.mean(-1).max(1)

    bound = H @ eps
    bound_ctx = np.einsum("ir,rc->ic", H, eps_ctx).max(1)
    identity = target - comp_pred - np.einsum("ir,rcnd->icnd", H, D)
    return {
        "losses": comp_loss,
        "epsilon": eps,
        "row_errors": rows,
        "row_bounds": bound,
        "row_bounds_context": bound_ctx,
        "coverage_error": float(rows.max()),
        "bound": float(bound.max()),
        "bound_clipped": float(min(1.0, bound.max())),
        "bound_context": float(bound_ctx.max()),
        "identity_max_abs": float(np.abs(identity).max()),
        "max_bound_excess": float((rows - bound).max()),
        "max_context_bound_excess": float((rows - bound_ctx).max()),
    }


# --------------------------------------------------------------------------
# near-optimal witness bracket (protocol section 3.8)
# --------------------------------------------------------------------------

def bracket_mass(P_fit: np.ndarray, target: int, donors: Sequence[int],
                 mass_on: Sequence[int], opt: float, slack: float,
                 senses: Sequence[str] = ("min", "max")) -> Dict[str, object]:
    """Min and max total weight on `mass_on` over witnesses within `slack` of `opt`.

    Exact LP over the FIT sample: per-pair slack t >= |residual_k| (TV equals
    the largest coordinate gap for three-class distributions), context means
    <= opt + slack, weights on the simplex over `donors`.  Returns both
    extreme weight vectors so the composition can be rerun at each end.
    """
    donors = list(donors)
    k = len(donors)
    X = P_fit[donors]                     # [k, C, n, 3]
    y = P_fit[target]                     # [C, n, 3]
    _, C, n, d = X.shape
    m = C * n
    feats = X.transpose(1, 2, 3, 0).reshape(m * d, k)
    t_cols = sparse.coo_matrix(
        (-np.ones(m * d), (np.arange(m * d), np.repeat(np.arange(m), d))), shape=(m * d, m)
    )
    up = sparse.hstack([sparse.csr_matrix(feats), t_cols])
    lo = sparse.hstack([sparse.csr_matrix(-feats), t_cols])
    avg = sparse.coo_matrix(
        (np.full(m, 1.0 / n), (np.repeat(np.arange(C), n), np.arange(m))), shape=(C, m)
    )
    ctx = sparse.hstack([sparse.csr_matrix((C, k)), avg])
    A_ub = sparse.vstack([up, lo, ctx], format="csr")
    flat = y.reshape(-1)
    b_ub = np.concatenate([flat, -flat, np.full(C, opt + slack)])
    A_eq = sparse.csr_matrix((np.ones(k), (np.zeros(k, int), np.arange(k))), shape=(1, k + m))
    sel = np.zeros(k + m)
    for j in mass_on:
        sel[donors.index(j)] = 1.0

    out: Dict[str, object] = {}
    for sense, sign in (("min", 1.0), ("max", -1.0)):
        if sense not in senses:
            continue
        res = linprog(sign * sel, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=(0, None), method="highs",
                      options={"primal_feasibility_tolerance": 1e-9,
                               "dual_feasibility_tolerance": 1e-9})
        if not res.success:
            raise RuntimeError(f"bracket LP ({sense}) failed for target {target}: {res.message}")
        w = np.clip(res.x[:k], 0.0, None)
        w /= w.sum()
        row = np.zeros(P_fit.shape[0])
        row[donors] = w
        pred = np.einsum("j,jcnd->cnd", row, P_fit)
        out[sense] = {
            "mass": float(sum(row[j] for j in mass_on)),
            "row": row,
            "fit_error": float(tv(y, pred).mean(-1).max()),
        }
    return out


# --------------------------------------------------------------------------
# paired source-task bootstrap
# --------------------------------------------------------------------------

def bootstrap_means(losses: np.ndarray, groups: np.ndarray, n_boot: int, seed: int,
                    batch: int = 500) -> np.ndarray:
    """Pair-weighted means under source-task resampling, paired across all rows.

    losses: [..., n] ; groups: [n] integer task ids.  Every leading row uses
    the same task multiplicities in a given draw.  Returns [n_boot, ...].
    """
    loss = np.asarray(losses, float)
    _, inv = np.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=G).astype(float)
    flat = loss.reshape(-1, loss.shape[-1])
    # task sums via a sparse indicator: [n, G]
    ind = sparse.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), G))
    gsum = np.asarray((ind.T @ flat.T)).T                 # [rows, G]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, flat.shape[0]))
    for s in range(0, n_boot, batch):
        b = min(batch, n_boot - s)
        mult = rng.multinomial(G, np.full(G, 1.0 / G), size=b).astype(float)
        # numpy 2.0 on Apple Accelerate raises spurious FP flags in matmul;
        # the values are verified finite below and by a hand-computed test.
        with np.errstate(all="ignore"):
            out[s:s + b] = (mult @ gsum.T) / (mult @ counts)[:, None]
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite bootstrap means")
    return out.reshape((n_boot,) + loss.shape[:-1])


def paired_comparison(base: np.ndarray, other: np.ndarray, groups: np.ndarray, gamma: float,
                      n_boot: int, seed: int, alpha: float = 0.05,
                      family_size: int = 1) -> List[Dict[str, object]]:
    """Compare two arms evaluated on the same pairs, e.g. individual vs joint.

    base, other: [T, C, n] losses of the same targets.  Rows: aggregate
    (target -1, the maximum over targets) then each target.  Gives paired
    percentile intervals for the difference of maxima, Bonferroni-adjusted
    intervals for the family, and the simultaneous sup-norm band of protocol
    section 3.4 projected onto each maximum.
    """
    stacked = np.stack([base, other])                     # [2, T, C, n]
    means = stacked.mean(-1)
    draws = bootstrap_means(stacked, groups, n_boot, seed)  # [B, 2, T, C]
    q = float(np.quantile(np.abs(draws - means[None]).max(axis=(1, 2, 3)),
                          1 - alpha / family_size))
    lo = np.clip(means - q, 0, 1)
    hi = np.clip(means + q, 0, 1)
    rows = []
    for t in [-1] + list(range(base.shape[0])):
        if t < 0:
            pt = means.max(axis=(1, 2))
            smp = draws.max(axis=(2, 3))
            blo, bhi = lo.max(axis=(1, 2)), hi.max(axis=(1, 2))
        else:
            pt = means[:, t].max(axis=1)
            smp = draws[:, :, t].max(axis=2)
            blo, bhi = lo[:, t].max(axis=1), hi[:, t].max(axis=1)
        delta = smp[:, 1] - smp[:, 0]
        # near-tie diagnostics: a max of means is non-smooth where the top
        # coordinates tie, so report how stable the maximiser is and how far
        # the resampled max sits above the point max.
        if t < 0:
            flat_pt = means.reshape(2, -1).argmax(1)
            flat_bs = draws.reshape(n_boot, 2, -1).argmax(2)
        else:
            flat_pt = means[:, t].argmax(1)
            flat_bs = draws[:, :, t].argmax(2)
        argmax_stab = (flat_bs == flat_pt[None]).mean(0)
        bias = smp.mean(0) - pt
        ci = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
        adj = np.quantile(delta, [alpha / (2 * family_size), 1 - alpha / (2 * family_size)])
        oci = np.quantile(smp[:, 1], [alpha / 2, 1 - alpha / 2])
        bci = np.quantile(smp[:, 0], [alpha / 2, 1 - alpha / 2])
        rows.append({
            "target": t,
            "base_error": float(pt[0]), "other_error": float(pt[1]),
            "delta": float(pt[1] - pt[0]),
            "delta_ci_low": float(ci[0]), "delta_ci_high": float(ci[1]),
            "delta_adj_ci_low": float(adj[0]), "delta_adj_ci_high": float(adj[1]),
            "frac_draws_delta_le_0": float((delta <= 0).mean()),
            "base_ci_low": float(bci[0]), "base_ci_high": float(bci[1]),
            "other_ci_low": float(oci[0]), "other_ci_high": float(oci[1]),
            "margin": float(pt[1] - gamma),
            "margin_ci_low": float(oci[0] - gamma), "margin_ci_high": float(oci[1] - gamma),
            "band_delta_low": float(blo[1] - bhi[0]), "band_delta_high": float(bhi[1] - blo[0]),
            "band_margin_low": float(blo[1] - gamma), "band_margin_high": float(bhi[1] - gamma),
            "band_radius": q, "family_size": family_size,
            "argmax_stability_base": float(argmax_stab[0]),
            "argmax_stability_other": float(argmax_stab[1]),
            "boot_bias_base": float(bias[0]), "boot_bias_other": float(bias[1]),
            "n_boot": n_boot, "seed": seed, "alpha": alpha,
        })
    return rows


def coordinate_deltas(base: np.ndarray, other: np.ndarray, groups: np.ndarray,
                      n_boot: int, seed: int, alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Per-(target, context) mean differences with a studentised max-t band.

    Each coordinate difference is a smooth mean contrast, so the bootstrap is
    valid for it.  The max-t quantile makes the band simultaneous over all
    T x C coordinates.  Returns arrays of shape [T, C].
    """
    d = other - base                                       # [T, C, n]
    point = d.mean(-1)
    draws = bootstrap_means(d, groups, n_boot, seed)       # [B, T, C]
    sd = draws.std(0, ddof=1)
    sd_safe = np.where(sd > 0, sd, np.inf)
    tmax = np.abs((draws - point[None]) / sd_safe[None]).max(axis=(1, 2))
    q = float(np.quantile(tmax, 1 - alpha))
    pci = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    return {"point": point, "sd": sd, "maxt_q": q,
            "band_low": point - q * sd, "band_high": point + q * sd,
            "ci_low": pci[0], "ci_high": pci[1]}
