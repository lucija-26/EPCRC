"""Reconstruction baselines (plan section 21.3).

Selection decides *which* judges stay physical; reconstruction decides how a
retired judge is rebuilt from them.  Section 21.3 asks for five rules, and the
point of the comparison is not only which fits best.  Unconstrained rules can
fit better while producing outputs that are not probability vectors at all, so
this module reports both the error and how often the reconstruction left the
simplex.

Every rule fits on FIT and is scored on a split it never saw::

    w = fit_weights(kind, fit, target, kept)
    row = score_weights(evaluate, target, kept, w)

Only ``simplex`` optimises the worst-context minimax objective; the others are
least-squares rules, which is exactly the ablation the plan describes.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import nnls

from epcrc.judge import (
    JudgeResponses,
    context_errors,
    solve_minimax_weights,
    total_variation,
)

__all__ = ["RECONSTRUCTORS", "fit_weights", "score_weights", "SIMPLEX_TOL"]

SIMPLEX_TOL = 1e-6

# Ridge is given its own regularisation grid and picks the value that minimises
# its FIT error, so the baseline is not handicapped by an arbitrary alpha.
_RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _design(
    responses: JudgeResponses,
    target: int,
    kept: Sequence[int],
) -> tuple:
    """Flatten to a plain least-squares problem ``A w ~= b``.

    A row is one (context, item, outcome) coordinate, so the three outcomes of
    an item are three separate equations.
    """
    kept = list(kept)
    A_parts, b_parts = [], []
    for block in responses.blocks:
        peers = block[:, kept, :].transpose(0, 2, 1).reshape(-1, len(kept))
        A_parts.append(peers)
        b_parts.append(block[:, target, :].reshape(-1))
    return np.vstack(A_parts), np.concatenate(b_parts)


def fit_weights(
    kind: str,
    fit: JudgeResponses,
    target: int,
    kept: Sequence[int],
) -> np.ndarray:
    """Fit one weight vector over `kept` that reconstructs `target` on FIT."""
    kept = list(kept)
    m = len(kept)
    if m == 0:
        raise ValueError("cannot reconstruct from an empty physical panel")

    # A retained judge always reconstructs itself exactly, whatever the rule.
    if target in kept:
        w = np.zeros(m)
        w[kept.index(target)] = 1.0
        return w

    if kind == "simplex":
        _, w = solve_minimax_weights(fit, target, kept)
        return w

    if kind == "uniform":
        return np.full(m, 1.0 / m)

    if kind == "nearest":
        distances = [
            float(context_errors(
                fit, target, [s], np.array([1.0])
            ).max())
            for s in kept
        ]
        w = np.zeros(m)
        w[int(np.argmin(distances))] = 1.0
        return w

    A, b = _design(fit, target, kept)

    if kind == "nonneg":
        w, _ = nnls(A, b)
        return w

    if kind == "ridge":
        best, best_err = None, np.inf
        gram = A.T @ A
        rhs = A.T @ b
        for alpha in _RIDGE_ALPHAS:
            w = np.linalg.solve(gram + alpha * np.eye(m), rhs)
            err = float(context_errors(fit, target, kept, w).max())
            if err < best_err:
                best, best_err = w, err
        return best

    raise ValueError(f"unknown reconstruction rule {kind!r}")


def score_weights(
    evaluate: JudgeResponses,
    target: int,
    kept: Sequence[int],
    weights: np.ndarray,
) -> Dict[str, float]:
    """Error of a fitted weight vector on a held-out split.

    ``off_simplex_frac`` is the share of reconstructed items that are not
    probability vectors -- negative mass or mass not summing to one.  It is the
    cost the unconstrained rules pay for their better fit, and it is why the
    plan keeps the simplex rule as the deployment certificate.
    """
    kept = list(kept)
    w = np.asarray(weights, dtype=float).reshape(-1)

    per_context, off_simplex, verdict_hits, n_rows = [], 0, 0, 0
    for block in evaluate.blocks:
        recon = np.tensordot(block[:, kept, :], w, axes=([1], [0]))
        truth = block[:, target, :]

        per_context.append(float(total_variation(truth, recon).mean()))
        off_simplex += int((
            (recon < -SIMPLEX_TOL).any(axis=1)
            | (np.abs(recon.sum(axis=1) - 1.0) > SIMPLEX_TOL)
        ).sum())
        verdict_hits += int((truth.argmax(axis=1) == recon.argmax(axis=1)).sum())
        n_rows += block.shape[0]

    return {
        "worst_context_mean_tv": float(max(per_context)),
        "mean_context_mean_tv": float(np.mean(per_context)),
        "verdict_agreement": verdict_hits / n_rows,
        "off_simplex_frac": off_simplex / n_rows,
        "weight_l1": float(np.abs(w).sum()),
        "weight_min": float(w.min()),
    }


RECONSTRUCTORS: List[str] = ["simplex", "nonneg", "ridge", "nearest", "uniform"]
