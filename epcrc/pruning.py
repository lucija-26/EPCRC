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
    sum_uniqueness: float
    action: str  # "add" | "remove" | "stop"


@dataclass
class PruningResult:
    kept_set: Set[int]
    coverage: float
    sum_uniqueness: float
    history: list[PruningStep]
    certificates: Dict[int, SubstitutionCertificate]


class ForwardSelectionPruner:
    """Section 4.2 forward selection.

    Start with S = empty and greedily add the model that reduces E(S) the most,
    until E(S) <= gamma.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
    ):
        self.coverage_fn = coverage_fn
        self.gamma = float(tolerance_gamma)

    def run(self, debug: bool = False) -> PruningResult:
        all_models = set(range(self.coverage_fn.N))
        S: Set[int] = set()
        history: list[PruningStep] = []
        it = 0

        while True:
            it += 1
            remaining = all_models - S

            if len(remaining) == 0:
                # All models added, nothing left to try
                E_now, certs_now = self.coverage_fn.compute_coverage(S, return_certificates=True)
                sum_u = self.coverage_fn.compute_sum_uniqueness(S)
                history.append(
                    PruningStep(
                        iteration=it,
                        removed_model_idx=None,
                        removed_model_name=None,
                        kept_set=set(S),
                        coverage=E_now,
                        sum_uniqueness=sum_u,
                        action="stop",
                    )
                )
                assert certs_now is not None
                return PruningResult(
                    kept_set=set(S),
                    coverage=E_now,
                    sum_uniqueness=sum_u,
                    history=history,
                    certificates=certs_now,
                )

            # Evaluate all possible single additions
            candidates: list[tuple[int, float]] = []
            for j in sorted(remaining):
                S_candidate = S | {j}
                E_candidate, _ = self.coverage_fn.compute_coverage(S_candidate)
                candidates.append((j, E_candidate))

            if debug:
                print(f"\n[Iteration {it}] Testing additions (γ = {self.gamma}):")
                for j, E_cand in sorted(candidates, key=lambda x: x[1]):
                    model_name = self.coverage_fn.model_names[j]
                    result_str = "✓" if E_cand <= self.gamma else "·"
                    print(f"  {result_str} {model_name:40s} → E(S) = {E_cand:.6f}")

            # Pick the model whose addition minimises E(S)
            best_j, best_coverage = min(candidates, key=lambda x: x[1])

            model_name = self.coverage_fn.model_names[best_j]
            S.add(best_j)
            sum_u = self.coverage_fn.compute_sum_uniqueness(S)
            history.append(
                PruningStep(
                    iteration=it,
                    removed_model_idx=best_j,       # here it means "added"
                    removed_model_name=model_name,
                    kept_set=set(S),
                    coverage=best_coverage,
                    sum_uniqueness=sum_u,
                    action="add",
                )
            )

            if best_coverage <= self.gamma:
                # Coverage satisfied — done
                E_now, certs_now = self.coverage_fn.compute_coverage(S, return_certificates=True)
                sum_u = self.coverage_fn.compute_sum_uniqueness(S)
                history.append(
                    PruningStep(
                        iteration=it + 1,
                        removed_model_idx=None,
                        removed_model_name=None,
                        kept_set=set(S),
                        coverage=E_now,
                        sum_uniqueness=sum_u,
                        action="stop",
                    )
                )
                assert certs_now is not None
                return PruningResult(
                    kept_set=set(S),
                    coverage=E_now,
                    sum_uniqueness=sum_u,
                    history=history,
                    certificates=certs_now,
                )


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
                sum_u = self.coverage_fn.compute_sum_uniqueness(S)
                history.append(
                    PruningStep(
                        iteration=it,
                        removed_model_idx=None,
                        removed_model_name=None,
                        kept_set=set(S),
                        coverage=E_now,
                        sum_uniqueness=sum_u,
                        action="stop",
                    )
                )
                assert certs_now is not None
                return PruningResult(
                    kept_set=set(S),
                    coverage=E_now,
                    sum_uniqueness=sum_u,
                    history=history,
                    certificates=certs_now,
                )

            S.remove(best_candidate)
            model_name = self.coverage_fn.model_names[best_candidate]
            sum_u = self.coverage_fn.compute_sum_uniqueness(S)
            history.append(
                PruningStep(
                    iteration=it,
                    removed_model_idx=best_candidate,
                    removed_model_name=model_name,
                    kept_set=set(S),
                    coverage=best_coverage,
                    sum_uniqueness=sum_u,
                    action="remove",
                )
            )
