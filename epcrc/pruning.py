from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

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


class PriorityQueuePruner:
    """Lazy greedy forward selection with priority queue, followed by backward cleanup.

    Phase 1 (forward / add): Use a max-heap with lazy re-evaluation to greedily
    add models that reduce E(S) the most, until E(S) <= gamma or no beneficial
    add exists.

    Phase 2 (backward / prune): Simple sweep to remove any model from S whose
    removal keeps E(S) <= gamma.
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
        history: List[PruningStep] = []
        it = 0

        # Current coverage of empty set
        E_current, _ = self.coverage_fn.compute_coverage(S)

        # --- Phase 1: Lazy greedy forward selection ---
        # Generation counter: incremented each time S changes.
        # Heap entries with generation < current are stale.
        generation = 0

        # Compute initial gains for all models and build max-heap.
        # Heap entries: (-gain, timestamp, model_idx, entry_generation)
        # Negate gain because heapq is a min-heap.
        heap: List[tuple] = []
        timestamp = 0  # tie-breaker for FIFO ordering
        for m in sorted(all_models):
            E_with_m, _ = self.coverage_fn.compute_coverage(S | {m})
            gain = E_current - E_with_m
            heapq.heappush(heap, (-gain, timestamp, m, generation))
            timestamp += 1

        if debug:
            print(f"[Phase 1] Initial E(empty) = {E_current:.6f}")
            print(f"[Phase 1] Initial heap size = {len(heap)}")
            for neg_g, _, m, _ in sorted(heap):
                name = self.coverage_fn.model_names[m]
                print(f"  g({name}) = {-neg_g:.6f}")

        phase1_done = False
        while heap and not phase1_done:
            neg_gain, _, m_idx, entry_gen = heapq.heappop(heap)
            gain = -neg_gain

            if m_idx in S:
                # Already added in a previous iteration, skip
                continue

            if entry_gen == generation:
                # Fresh entry — decide whether to accept
                if gain > 0:
                    # Accept: add model to S
                    it += 1
                    S.add(m_idx)
                    E_current, _ = self.coverage_fn.compute_coverage(S)
                    generation += 1  # all remaining entries are now stale

                    model_name = self.coverage_fn.model_names[m_idx]
                    sum_u = self.coverage_fn.compute_sum_uniqueness(S)
                    history.append(
                        PruningStep(
                            iteration=it,
                            removed_model_idx=m_idx,
                            removed_model_name=model_name,
                            kept_set=set(S),
                            coverage=E_current,
                            sum_uniqueness=sum_u,
                            action="add",
                        )
                    )

                    if debug:
                        print(
                            f"[Phase 1] iter {it}: ADD {model_name}  "
                            f"gain={gain:.6f}  E(S)={E_current:.6f}  |S|={len(S)}"
                        )

                    if E_current <= self.gamma:
                        # Coverage satisfied — exit phase 1
                        if debug:
                            print(f"[Phase 1] E(S)={E_current:.6f} <= gamma={self.gamma} -> DONE")
                        phase1_done = True
                else:
                    # Best fresh gain is non-positive — no beneficial add exists
                    if debug:
                        print(
                            f"[Phase 1] Best gain={gain:.6f} <= 0 for "
                            f"{self.coverage_fn.model_names[m_idx]} -> STOP"
                        )
                    phase1_done = True
            else:
                # Stale entry — recompute gain with current S
                E_with_m, _ = self.coverage_fn.compute_coverage(S | {m_idx})
                new_gain = E_current - E_with_m
                heapq.heappush(heap, (-new_gain, timestamp, m_idx, generation))
                timestamp += 1

        # --- Phase 2: Simple backward pruning ---
        if debug:
            print(f"\n[Phase 2] Starting backward cleanup, |S|={len(S)}, E(S)={E_current:.6f}")

        changed = True
        while changed:
            changed = False
            for m_idx in sorted(S):
                S_without = S - {m_idx}
                E_without, _ = self.coverage_fn.compute_coverage(S_without)
                if E_without <= self.gamma:
                    it += 1
                    S = S_without
                    E_current = E_without
                    model_name = self.coverage_fn.model_names[m_idx]
                    sum_u = self.coverage_fn.compute_sum_uniqueness(S)
                    history.append(
                        PruningStep(
                            iteration=it,
                            removed_model_idx=m_idx,
                            removed_model_name=model_name,
                            kept_set=set(S),
                            coverage=E_current,
                            sum_uniqueness=sum_u,
                            action="remove",
                        )
                    )

                    if debug:
                        print(
                            f"[Phase 2] iter {it}: REMOVE {model_name}  "
                            f"E(S)={E_current:.6f}  |S|={len(S)}"
                        )
                    changed = True
                    break  # restart the inner loop with updated S

        # --- Final stop step ---
        it += 1
        E_final, certs_final = self.coverage_fn.compute_coverage(S, return_certificates=True)
        sum_u = self.coverage_fn.compute_sum_uniqueness(S)
        history.append(
            PruningStep(
                iteration=it,
                removed_model_idx=None,
                removed_model_name=None,
                kept_set=set(S),
                coverage=E_final,
                sum_uniqueness=sum_u,
                action="stop",
            )
        )
        assert certs_final is not None
        return PruningResult(
            kept_set=set(S),
            coverage=E_final,
            sum_uniqueness=sum_u,
            history=history,
            certificates=certs_final,
        )
