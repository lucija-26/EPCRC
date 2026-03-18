from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

from .coverage import CoverageFunctional, SubstitutionCertificate


@dataclass
class PruningStep:
    iteration: int
    removed_model_idx: Optional[int]
    removed_model_name: Optional[str]
    kept_set: Set[int]
    coverage: float
    action: str  # "remove" | "stop"


@dataclass
class PruningResult:
    kept_set: Set[int]
    coverage: float
    history: list[PruningStep]
    certificates: Dict[int, SubstitutionCertificate]


class BackwardEliminationPruner:
    """Section 4.1 backward elimination.

    Start with S = J and remove one model at a time if coverage remains <= gamma.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
    ):
        self.coverage_fn = coverage_fn
        self.gamma = float(tolerance_gamma)

    def run(self, debug: bool = False) -> PruningResult:
        S = set(range(self.coverage_fn.N))
        history: list[PruningStep] = []
        it = 0

        while True:
            it += 1
            
            # Evaluate all possible single removals
            candidates: list[tuple[int, float]] = []  # (model_idx, E(S\{j}))
            for j in sorted(S):
                S_candidate = set(S)
                S_candidate.remove(j)
                E_candidate, _ = self.coverage_fn.compute_coverage(S_candidate)
                candidates.append((j, E_candidate))

            # If debug mode, print all candidates
            if debug:
                print(f"\n[Iteration {it}] Testing removals (γ = {self.gamma}):")
                for j, E_cand in sorted(candidates, key=lambda x: x[1]):
                    model_name = self.coverage_fn.model_names[j]
                    result_str = "✓" if E_cand <= self.gamma else "✗"
                    print(f"  {result_str} {model_name:40s} → E(S) = {E_cand:.6f}")

            # Find best candidate that passes threshold
            best_candidate = None
            best_coverage = float("inf")

            for j, E_candidate in candidates:
                if E_candidate <= self.gamma and E_candidate < best_coverage:
                    best_coverage = E_candidate
                    best_candidate = j

            if best_candidate is None:
                if debug:
                    print(f"  ⛔ STOP: All removals fail\n")
                E_now, certs_now = self.coverage_fn.compute_coverage(S, return_certificates=True)
                history.append(
                    PruningStep(
                        iteration=it,
                        removed_model_idx=None,
                        removed_model_name=None,
                        kept_set=set(S),
                        coverage=E_now,
                        action="stop",
                    )
                )
                assert certs_now is not None
                return PruningResult(
                    kept_set=set(S),
                    coverage=E_now,
                    history=history,
                    certificates=certs_now,
                )

            S.remove(best_candidate)
            model_name = self.coverage_fn.model_names[best_candidate]
            history.append(
                PruningStep(
                    iteration=it,
                    removed_model_idx=best_candidate,
                    removed_model_name=model_name,
                    kept_set=set(S),
                    coverage=best_coverage,
                    action="remove",
                )
            )
