from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import numpy as np


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

        w = cp.Variable(p)
        objective = cp.Minimize(cp.sum_squares(target - peers @ w))
        constraints = [w >= 0, cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)

        for solver in (cp.ECOS, cp.SCS, None):
            try:
                if solver is None:
                    problem.solve()
                else:
                    problem.solve(solver=solver)
                if w.value is not None:
                    break
            except Exception:
                continue

        if w.value is None:
            fallback = np.ones(p, dtype=float) / p
            dist = float(np.linalg.norm(target - peers @ fallback, ord=2))
            return dist, fallback

        weights = np.asarray(w.value, dtype=float).reshape(-1)
        weights = np.clip(weights, 0.0, None)
        s = float(weights.sum())
        if s <= 0:
            weights = np.ones(p, dtype=float) / p
        else:
            weights = weights / s

        dist = float(np.linalg.norm(target - peers @ weights, ord=2))
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

        pred = peers @ w
        residual = target - pred

        if metric == "mean_abs":
            return float(np.mean(np.abs(residual)))
        if metric == "rmse":
            return float(np.sqrt(np.mean(residual**2)))
        if metric == "max":
            return float(np.max(np.abs(residual)))

        raise ValueError(f"Unknown metric: {metric}")
