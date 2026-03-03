# epcrc/coverage.py
"""
Coverage functional for ecosystem pruning.

Implements the ecosystem coverage error E(S) from Section 2.4 of the paper:
    E(S) = max_{i ∈ J} U(i | S)

where U(i|S) is the uniqueness of model i relative to kept set S.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from epcrc.geometry import DISCOSolver


@dataclass
class SubstitutionCertificate:
    """
    Certificate proving a model can be substituted by convex routing.
    
    Contains:
    - The model being substituted
    - The kept set used for substitution
    - The routing weights (simplex)
    - The approximation error
    """
    model_idx: int
    model_name: str
    kept_set: Set[int]
    weights: np.ndarray  # Simplex weights over kept set
    uniqueness: float    # U(i|S) - residual magnitude
    
    def is_redundant(self, tolerance: float) -> bool:
        """Check if model is redundant within tolerance."""
        return self.uniqueness <= tolerance


class CoverageFunctional:
    """
    Computes ecosystem coverage error for a candidate kept set S.
    
    This is the core object for pruning decisions:
    - E(S) ≤ γ means S can substitute the entire ecosystem within tolerance γ
    - E(S) is monotone non-increasing in S (adding models can only help)
    """
    
    def __init__(
        self,
        Y_fit: np.ndarray,
        Y_eval: np.ndarray,
        model_names: Optional[List[str]] = None,
        metric: str = "mean_abs",
    ):
        """
        Args:
            Y_fit: Response matrix on fitting data, shape (n_fit, N_models)
            Y_eval: Response matrix on evaluation data, shape (n_eval, N_models)
            model_names: Optional names for each model
            metric: Uniqueness metric ("mean_abs", "rmse", "max")
        """
        self.Y_fit = np.asarray(Y_fit)
        self.Y_eval = np.asarray(Y_eval)
        self.n_fit, self.N = self.Y_fit.shape
        self.n_eval, N2 = self.Y_eval.shape
        
        if N2 != self.N:
            raise ValueError(
                f"Y_fit has {self.N} models but Y_eval has {N2}"
            )
        
        self.model_names = model_names or [f"model_{i}" for i in range(self.N)]
        self.metric = metric
        
        # Cache for computed certificates
        self._cache: Dict[frozenset, Dict[int, SubstitutionCertificate]] = {}
    
    def compute_certificate(
        self,
        target_idx: int,
        kept_set: Set[int],
    ) -> SubstitutionCertificate:
        """
        Compute substitution certificate for one model.
        
        Implements Equations 4-6 from the paper:
        1. Fit weights w* on P_fit
        2. Evaluate uniqueness U(i|S) on P_eval
        
        Args:
            target_idx: Index of model to be substituted
            kept_set: Set of model indices in S
            
        Returns:
            SubstitutionCertificate with weights and uniqueness
        """
        kept_list = sorted(kept_set)
        
        if target_idx in kept_set:
            # Model can substitute itself perfectly
            weights = np.zeros(len(kept_list))
            weights[kept_list.index(target_idx)] = 1.0
            return SubstitutionCertificate(
                model_idx=target_idx,
                model_name=self.model_names[target_idx],
                kept_set=kept_set,
                weights=weights,
                uniqueness=0.0,
            )
        
        if len(kept_set) == 0:
            return SubstitutionCertificate(
                model_idx=target_idx,
                model_name=self.model_names[target_idx],
                kept_set=kept_set,
                weights=np.array([]),
                uniqueness=float('inf'),
            )
        
        # Extract target and peer responses
        y_target_fit = self.Y_fit[:, target_idx]
        Y_peers_fit = self.Y_fit[:, kept_list]
        
        y_target_eval = self.Y_eval[:, target_idx]
        Y_peers_eval = self.Y_eval[:, kept_list]
        
        # Fit weights on P_fit
        _, weights = DISCOSolver.solve_weights_and_distance(
            target_vec=y_target_fit,
            peer_matrix=Y_peers_fit,
        )
        
        # Evaluate on P_eval
        uniqueness = DISCOSolver.compute_uniqueness(
            target_vec=y_target_eval,
            peer_matrix=Y_peers_eval,
            weights=weights,
            metric=self.metric,
        )
        
        return SubstitutionCertificate(
            model_idx=target_idx,
            model_name=self.model_names[target_idx],
            kept_set=kept_set,
            weights=weights,
            uniqueness=uniqueness,
        )
    
    def compute_coverage(
        self,
        kept_set: Set[int],
        return_certificates: bool = False,
    ) -> Tuple[float, Optional[Dict[int, SubstitutionCertificate]]]:
        """
        Compute ecosystem coverage error E(S).
        
        E(S) = max_{i ∈ J} U(i | S)
        
        This is Equation 7 from the paper.
        
        Args:
            kept_set: Set of model indices to keep (S)
            return_certificates: If True, also return all certificates
            
        Returns:
            coverage: E(S) value
            certificates: Dict mapping model_idx -> certificate (if requested)
        """
        kept_set = set(kept_set)
        cache_key = frozenset(kept_set)
        
        # Check cache
        if cache_key in self._cache:
            certificates = self._cache[cache_key]
            coverage = max(c.uniqueness for c in certificates.values())
            if return_certificates:
                return coverage, certificates
            return coverage, None
        
        # Compute all certificates
        certificates = {}
        for i in range(self.N):
            cert = self.compute_certificate(i, kept_set)
            certificates[i] = cert
        
        # Cache results
        self._cache[cache_key] = certificates
        
        # Coverage = max uniqueness over all models
        coverage = max(c.uniqueness for c in certificates.values())
        
        if return_certificates:
            return coverage, certificates
        return coverage, None
    
    def find_bottleneck(self, kept_set: Set[int]) -> SubstitutionCertificate:
        """
        Find the model that is hardest to substitute (bottleneck).
        
        This is the model that determines E(S).
        
        Args:
            kept_set: Current kept set S
            
        Returns:
            Certificate of the model with maximum uniqueness
        """
        _, certificates = self.compute_coverage(kept_set, return_certificates=True)
        return max(certificates.values(), key=lambda c: c.uniqueness)
    
    def clear_cache(self):
        """Clear the certificate cache."""
        self._cache.clear()
