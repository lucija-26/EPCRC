# epcrc/pruning.py
"""
Pruning algorithms for ecosystem consolidation.

Implements the algorithms from Section 4 of the paper:
- Backward elimination: Start full, remove redundant models
- Forward selection: Start empty, add necessary models
- Risk-controlled variants with UCB
"""

from typing import List, Set, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np

from epcrc.coverage import CoverageFunctional, SubstitutionCertificate


@dataclass
class PruningResult:
    """
    Result of a pruning algorithm.
    
    Contains:
    - Final kept set S*
    - Pruning path (sequence of sets visited)
    - Final coverage error
    - Substitution certificates for removed models
    """
    kept_set: Set[int]
    kept_names: List[str]
    removed_set: Set[int]
    removed_names: List[str]
    final_coverage: float
    path: List[Tuple[Set[int], float]]  # [(S, E(S)), ...]
    certificates: dict  # {removed_idx: certificate}


class BackwardElimination:
    """
    Backward elimination pruning (Section 4.1).
    
    Start from full ecosystem, remove models one by one.
    Remove model j if E(S \ {j}) ≤ γ.
    
    This is the simplest pruning algorithm. It removes models
    that are already inside the convex hull of remaining models.
    """
    
    def __init__(
        self,
        coverage: CoverageFunctional,
        tolerance: float,
        verbose: bool = True,
    ):
        """
        Args:
            coverage: CoverageFunctional object with response data
            tolerance: Maximum allowed coverage error γ
            verbose: Print progress
        """
        self.coverage = coverage
        self.tolerance = tolerance
        self.verbose = verbose
    
    def run(self) -> PruningResult:
        """
        Run backward elimination.
        
        Returns:
            PruningResult with final kept set and pruning path
        """
        N = self.coverage.N
        model_names = self.coverage.model_names
        
        # Start with full set
        S = set(range(N))
        path = []
        removed_certs = {}
        
        # Initial coverage
        E_S, certs = self.coverage.compute_coverage(S, return_certificates=True)
        path.append((S.copy(), E_S))
        
        if self.verbose:
            print(f"Initial: |S| = {len(S)}, E(S) = {E_S:.4f}")
        
        # Try removing each model
        changed = True
        while changed and len(S) > 1:
            changed = False
            
            for j in list(S):
                S_prime = S - {j}
                E_prime, _ = self.coverage.compute_coverage(S_prime)
                
                if E_prime <= self.tolerance:
                    # Accept removal
                    cert = self.coverage.compute_certificate(j, S_prime)
                    removed_certs[j] = cert
                    S = S_prime
                    path.append((S.copy(), E_prime))
                    changed = True
                    
                    if self.verbose:
                        print(
                            f"Removed {model_names[j]}: "
                            f"|S| = {len(S)}, E(S) = {E_prime:.4f}"
                        )
                    break
        
        final_E, _ = self.coverage.compute_coverage(S)
        removed = set(range(N)) - S
        
        return PruningResult(
            kept_set=S,
            kept_names=[model_names[i] for i in sorted(S)],
            removed_set=removed,
            removed_names=[model_names[i] for i in sorted(removed)],
            final_coverage=final_E,
            path=path,
            certificates=removed_certs,
        )


class ForwardSelection:
    """
    Forward selection pruning (Section 4.2).
    
    Start from empty set, add models that reduce E(S) the most.
    Stop when E(S) ≤ γ.
    
    Often yields smaller sets than backward elimination but is
    more computationally expensive.
    """
    
    def __init__(
        self,
        coverage: CoverageFunctional,
        tolerance: float,
        verbose: bool = True,
    ):
        """
        Args:
            coverage: CoverageFunctional object with response data
            tolerance: Target coverage error γ
            verbose: Print progress
        """
        self.coverage = coverage
        self.tolerance = tolerance
        self.verbose = verbose
    
    def run(self) -> PruningResult:
        """
        Run forward selection.
        
        Returns:
            PruningResult with final kept set and pruning path
        """
        N = self.coverage.N
        model_names = self.coverage.model_names
        
        # Start empty
        S = set()
        path = []
        
        if self.verbose:
            print(f"Target tolerance: γ = {self.tolerance:.4f}")
        
        while len(S) < N:
            # Find model that reduces coverage the most
            best_j = None
            best_E = float('inf')
            
            for j in range(N):
                if j in S:
                    continue
                
                S_plus = S | {j}
                E_plus, _ = self.coverage.compute_coverage(S_plus)
                
                if E_plus < best_E:
                    best_E = E_plus
                    best_j = j
            
            if best_j is None:
                break
            
            # Add best model
            S.add(best_j)
            path.append((S.copy(), best_E))
            
            if self.verbose:
                print(
                    f"Added {model_names[best_j]}: "
                    f"|S| = {len(S)}, E(S) = {best_E:.4f}"
                )
            
            # Check if we've reached tolerance
            if best_E <= self.tolerance:
                if self.verbose:
                    print(f"Reached tolerance with {len(S)} models")
                break
        
        final_E, certs = self.coverage.compute_coverage(S, return_certificates=True)
        removed = set(range(N)) - S
        
        # Build certificates for removed models
        removed_certs = {i: certs[i] for i in removed}
        
        return PruningResult(
            kept_set=S,
            kept_names=[model_names[i] for i in sorted(S)],
            removed_set=removed,
            removed_names=[model_names[i] for i in sorted(removed)],
            final_coverage=final_E,
            path=path,
            certificates=removed_certs,
        )


class GreedyPruning:
    """
    Greedy coverage-based pruning.
    
    Combines backward and forward ideas:
    - Remove the model with smallest uniqueness at each step
    - Stop when coverage exceeds tolerance
    
    This is often the most effective simple baseline.
    """
    
    def __init__(
        self,
        coverage: CoverageFunctional,
        tolerance: float,
        verbose: bool = True,
    ):
        self.coverage = coverage
        self.tolerance = tolerance
        self.verbose = verbose
    
    def run(self) -> PruningResult:
        """
        Run greedy pruning.
        
        At each step, remove the model that increases E(S) the least.
        """
        N = self.coverage.N
        model_names = self.coverage.model_names
        
        S = set(range(N))
        path = []
        removed_certs = {}
        
        E_S, _ = self.coverage.compute_coverage(S)
        path.append((S.copy(), E_S))
        
        if self.verbose:
            print(f"Initial: |S| = {len(S)}, E(S) = {E_S:.4f}")
        
        while len(S) > 1:
            # Find model whose removal increases coverage the least
            best_j = None
            best_E = float('inf')
            
            for j in S:
                S_minus = S - {j}
                E_minus, _ = self.coverage.compute_coverage(S_minus)
                
                if E_minus < best_E:
                    best_E = E_minus
                    best_j = j
            
            # Check if removal is acceptable
            if best_E > self.tolerance:
                if self.verbose:
                    print(
                        f"Stopping: removing any model would exceed γ={self.tolerance}"
                    )
                break
            
            # Accept removal
            cert = self.coverage.compute_certificate(best_j, S - {best_j})
            removed_certs[best_j] = cert
            S.remove(best_j)
            path.append((S.copy(), best_E))
            
            if self.verbose:
                print(
                    f"Removed {model_names[best_j]} (U={cert.uniqueness:.4f}): "
                    f"|S| = {len(S)}, E(S) = {best_E:.4f}"
                )
        
        final_E, _ = self.coverage.compute_coverage(S)
        removed = set(range(N)) - S
        
        return PruningResult(
            kept_set=S,
            kept_names=[model_names[i] for i in sorted(S)],
            removed_set=removed,
            removed_names=[model_names[i] for i in sorted(removed)],
            final_coverage=final_E,
            path=path,
            certificates=removed_certs,
        )


# =============================================================================
# Budget-constrained variants (Equation 9)
# =============================================================================

def prune_to_budget(
    coverage: CoverageFunctional,
    budget: int,
    method: str = "forward",
    verbose: bool = True,
) -> PruningResult:
    """
    Solve the budget-constrained problem (Equation 9):
        min_{S} E(S)  s.t. |S| ≤ k
    
    Args:
        coverage: CoverageFunctional
        budget: Maximum number of models k
        method: "forward" or "greedy"
        verbose: Print progress
        
    Returns:
        PruningResult
    """
    N = coverage.N
    
    if method == "forward":
        # Forward selection up to budget
        pruner = ForwardSelection(coverage, tolerance=float('inf'), verbose=verbose)
        result = pruner.run()
        
        # Trim to budget if needed
        if len(result.kept_set) > budget:
            # Re-run with budget constraint
            S = set()
            model_names = coverage.model_names
            
            for _ in range(budget):
                best_j, best_E = None, float('inf')
                for j in range(N):
                    if j in S:
                        continue
                    E, _ = coverage.compute_coverage(S | {j})
                    if E < best_E:
                        best_E, best_j = E, j
                if best_j is not None:
                    S.add(best_j)
            
            final_E, certs = coverage.compute_coverage(S, return_certificates=True)
            removed = set(range(N)) - S
            
            return PruningResult(
                kept_set=S,
                kept_names=[model_names[i] for i in sorted(S)],
                removed_set=removed,
                removed_names=[model_names[i] for i in sorted(removed)],
                final_coverage=final_E,
                path=[(S, final_E)],
                certificates={i: certs[i] for i in removed},
            )
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}")
