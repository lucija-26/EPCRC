from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize, nnls


def suppress_spurious_blas_flags() -> np.errstate:
    """Mask divide/overflow/invalid flags raised by the BLAS matmul kernel.

    numpy >= 2.0 reports these on essentially every `peers @ w` in this project
    even though the inputs are finite and the result is exact to ~1e-16 -- they
    come from unused SIMD lanes in the vectorised kernel, not from the data.
    Left unsuppressed a single gamma sweep emits tens of MB of stderr, which
    hides genuine warnings.  Callers must still check their own results are
    finite; this only silences the flag, it does not make a real blow-up safe.
    """
    return np.errstate(over="ignore", divide="ignore", invalid="ignore")


def _slsqp_simplex(target: np.ndarray, peers: np.ndarray) -> np.ndarray:
    """Solve min ||target - peers @ w||^2 s.t. w >= 0, sum(w) = 1 via SLSQP.

    Only sound on unit-scale inputs (callers must normalize first); used as a
    fallback when NNLS fails to converge.
    """
    p = peers.shape[1]
    PtP = peers.T @ peers
    Pty = peers.T @ target

    def obj_and_grad(w):
        return float(0.5 * w @ PtP @ w - Pty @ w), PtP @ w - Pty

    result = minimize(
        obj_and_grad, np.ones(p) / p, jac=True,
        method="SLSQP", bounds=[(0.0, None)] * p,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                     "jac": lambda w: np.ones(p)},
        options={"maxiter": 500, "ftol": 1e-14},
    )
    return np.asarray(result.x, dtype=float)


class DISCOSolver:
    """Simplex-constrained projection utilities.

    For target vector y and peer matrix P, solve:
        min_w ||y - P w||_2^2
        s.t. w >= 0, sum(w) = 1
    """

    @staticmethod
    def solve_weights_and_distance(
        target_vec: np.ndarray,
        peer_matrix: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        target = np.asarray(target_vec, dtype=float).reshape(-1)
        peers = np.asarray(peer_matrix, dtype=float)

        if peers.ndim == 1:
            peers = peers.reshape(-1, 1)

        n, p = peers.shape
        if target.shape[0] != n:
            raise ValueError(
                f"target length {target.shape[0]} does not match peer rows {n}"
            )

        if p == 0:
            return float("inf"), np.array([], dtype=float)

        if p == 1:
            weights = np.array([1.0])
            dist = float(np.linalg.norm(target - peers @ weights, ord=2))
            return dist, weights

        # Normalize to unit scale: the solve is scale-invariant in exact
        # arithmetic, but unscaled traffic data (values ~5e3, objective ~1e10)
        # previously made the optimizer terminate far from the optimum.
        scale = max(float(np.abs(peers).max()), float(np.abs(target).max()))
        if scale <= 0:
            weights = np.ones(p, dtype=float) / p
            return 0.0, weights

        # Simplex-constrained least squares via penalty-augmented NNLS:
        # append a row lam * 1^T w = lam so NNLS drives sum(w) -> 1, then
        # renormalize to enforce the constraint exactly.  lam is large enough
        # that the constraint violation is ~1e-9 on unit-scale data, and small
        # enough not to swamp the data block in the normal equations.
        lam = 1e4
        A = np.vstack([peers / scale, lam * np.ones((1, p))])
        b = np.concatenate([target / scale, [lam]])

        with suppress_spurious_blas_flags():
            try:
                w, _ = nnls(A, b, maxiter=100 * p)
            except RuntimeError:
                # NNLS's active-set method can cycle on degenerate/underdetermined
                # systems; fall back to SLSQP on the normalized problem (correct
                # there, unlike on raw-scale data -- just slower).
                w = _slsqp_simplex(target / scale, peers / scale)

            weights = np.clip(w, 0.0, None)
            s = float(weights.sum())
            if s <= 0:
                weights = np.ones(p, dtype=float) / p
            else:
                weights = weights / s

            dist = float(np.linalg.norm(target - peers @ weights, ord=2))

        if not (np.isfinite(dist) and np.isfinite(weights).all()):
            raise FloatingPointError(
                f"simplex projection diverged: p={p}, scale={scale:.3g}, dist={dist}"
            )
        return dist, weights

    @staticmethod
    def compute_uniqueness(
        target_vec: np.ndarray,
        peer_matrix: np.ndarray,
        weights: np.ndarray,
        metric: str = "mean_abs",
    ) -> float:
        target = np.asarray(target_vec, dtype=float).reshape(-1)
        peers = np.asarray(peer_matrix, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)

        if peers.ndim == 1:
            peers = peers.reshape(-1, 1)

        with suppress_spurious_blas_flags():
            residual = target - peers @ w

        if metric == "mean_abs":
            return float(np.mean(np.abs(residual)))
        if metric == "rmse":
            return float(np.sqrt(np.mean(residual**2)))
        if metric == "max":
            return float(np.max(np.abs(residual)))

        raise ValueError(f"Unknown metric: {metric}")
