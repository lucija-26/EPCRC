# epcrc/geometry.py
"""
Convex geometry solver for EPCRC.

Implements the DISCO-style simplex-constrained projection:
    w* = argmin_{w ∈ Δ} ||y_target - Φ_S @ w||_2
    
This finds the best convex combination of peer models to approximate
a target model's behavior.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional


class DISCOSolver:
    """
    Solves the convex projection problem for routing certificates.
    
    Given a target response vector and peer response matrix, finds
    simplex weights that minimize approximation error.
    
    This is the core operation for:
    - Computing substitution certificates (w*)
    - Measuring peer-inexpressible residual (PIER / uniqueness)
    - Evaluating ecosystem coverage
    """
    
    @staticmethod
    def solve_weights_and_distance(
        target_vec: np.ndarray,
        peer_matrix: np.ndarray,
        solver: str = "ECOS",
    ) -> Tuple[float, np.ndarray]:
        """
        Solve the simplex-constrained least squares projection.
        
        Minimizes: ||target - peers @ w||_2
        Subject to: w ∈ Δ^{|S|-1} (simplex)
        
        This implements Equation 4 from the paper.
        
        Args:
            target_vec: Response vector of target model, shape (D,) or (D, 1)
            peer_matrix: Response matrix of peers, shape (D, N_peers)
            solver: CVXPY solver to use ("ECOS", "SCS", or "auto")
            
        Returns:
            distance: L2 distance to convex hull (the residual norm)
            weights: Optimal simplex weights w*, shape (N_peers,)
        """
        # Ensure target is 1D
        target_vec = np.asarray(target_vec).flatten()
        peer_matrix = np.asarray(peer_matrix)
        
        # Handle edge cases
        if peer_matrix.ndim == 1:
            peer_matrix = peer_matrix.reshape(-1, 1)
        
        D, N_peers = peer_matrix.shape
        
        if len(target_vec) != D:
            raise ValueError(
                f"Shape mismatch: target has {len(target_vec)} samples, "
                f"peers have {D} samples"
            )
        
        if N_peers == 0:
            return float('inf'), np.array([])
        
        # Define optimization variable
        w = cp.Variable(N_peers)
        
        # Objective: minimize L2 distance
        objective = cp.Minimize(cp.norm(target_vec - peer_matrix @ w, 2))
        
        # Constraints: simplex (non-negative, sum to 1)
        constraints = [
            w >= 0,
            cp.sum(w) == 1,
        ]
        
        # Solve
        prob = cp.Problem(objective, constraints)
        
        try:
            if solver.upper() == "ECOS":
                prob.solve(solver=cp.ECOS)
            elif solver.upper() == "SCS":
                prob.solve(solver=cp.SCS)
            else:
                prob.solve()
        except Exception:
            # Fallback solver chain
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                prob.solve()
        
        if w.value is None:
            # Solver failed - return uniform weights as fallback
            return float('inf'), np.ones(N_peers) / N_peers
        
        return float(prob.value), np.array(w.value)
    
    @staticmethod
    def compute_residual(
        target_vec: np.ndarray,
        peer_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the peer-inexpressible residual vector.
        
        R_{i|S}(x, θ) = Y_i(x, θ) - w*ᵀ Φ_S(x, θ)
        
        This is Equation 5 from the paper.
        
        Args:
            target_vec: Target response vector, shape (D,)
            peer_matrix: Peer response matrix, shape (D, N_peers)
            weights: Simplex weights, shape (N_peers,)
            
        Returns:
            Residual vector, shape (D,)
        """
        target_vec = np.asarray(target_vec).flatten()
        weights = np.asarray(weights).flatten()
        
        # Convex combination: Φ_S @ w
        y_hat = peer_matrix @ weights
        
        return target_vec - y_hat
    
    @staticmethod
    def compute_uniqueness(
        target_vec: np.ndarray,
        peer_matrix: np.ndarray,
        weights: np.ndarray,
        metric: str = "mean_abs",
    ) -> float:
        """
        Compute uniqueness metric U(i|S) from residuals.
        
        U(i|S) = E_{P_eval}[|R_{i|S}(X, θ)|]
        
        This is Equation 6 from the paper.
        
        Args:
            target_vec: Target response vector
            peer_matrix: Peer response matrix
            weights: Simplex weights
            metric: "mean_abs" (L1), "rmse" (L2), or "max"
            
        Returns:
            Uniqueness score (smaller = more redundant)
        """
        residual = DISCOSolver.compute_residual(target_vec, peer_matrix, weights)
        
        if metric == "mean_abs":
            return float(np.mean(np.abs(residual)))
        elif metric == "rmse":
            return float(np.sqrt(np.mean(residual ** 2)))
        elif metric == "max":
            return float(np.max(np.abs(residual)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
