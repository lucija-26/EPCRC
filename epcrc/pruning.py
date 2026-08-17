from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from .coverage import CoverageFunctional, SubstitutionCertificate

logger = logging.getLogger(__name__)


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

    cleanup=False is the literal paper-4.2 rule: stop at the first feasible
    prefix.  Because E is non-monotone, a model added early can be made
    redundant by later additions, so that prefix is typically not even locally
    minimal -- on UTD19 it leaves up to 5 droppable models.

    cleanup=True (default) additionally trims, using the same lowest-E single
    deletion rule as BackwardEliminationPruner, so the result ends on a
    comparable single-deletion optimum.  Note this makes it a forward/backward
    hybrid rather than forward selection: backward elimination beats the
    cleanup=False variant, and only the trimmed variant beats backward.  Report
    the two separately (run_all.py calls them forward and forward_trim) instead
    of collapsing them into one "forward" number.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
        cleanup: bool = True,
    ):
        self.coverage_fn = coverage_fn
        self.gamma = float(tolerance_gamma)
        self.cleanup = bool(cleanup)

    def _finalize(
        self,
        S: Set[int],
        history: list[PruningStep],
        it: int,
        debug: bool = False,
    ) -> PruningResult:
        """Trim redundant models, then package the result."""
        if self.cleanup:
            while len(S) > 1:
                # Same rule as BackwardEliminationPruner: drop the removal that
                # leaves coverage lowest, so both methods end on a comparable
                # single-deletion optimum.
                best_j, best_E = None, float("inf")
                for j in sorted(S):
                    E_cand, _ = self.coverage_fn.compute_coverage(S - {j})
                    if E_cand <= self.gamma and E_cand < best_E:
                        best_j, best_E = j, E_cand
                if best_j is None:
                    break
                S = S - {best_j}
                it += 1
                if debug:
                    print(f"[cleanup] REMOVE {self.coverage_fn.model_names[best_j]} "
                          f"-> E(S)={best_E:.6f}  |S|={len(S)}")
                history.append(
                    PruningStep(
                        iteration=it,
                        removed_model_idx=best_j,
                        removed_model_name=self.coverage_fn.model_names[best_j],
                        kept_set=set(S),
                        coverage=best_E,
                        sum_uniqueness=self.coverage_fn.compute_sum_uniqueness(S),
                        action="remove",
                    )
                )

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
                return self._finalize(S, history, it, debug)

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
                return self._finalize(S, history, it, debug)


class WarmStartForwardWorstCoveredPruner:
    """Warm-started forward pruning seeded with an initial kept set.

    Starts from a user-provided seed set M (instead of the empty set), then
    repeatedly adds the currently worst-covered model
        argmax_i U(i | S)
    until E(S) <= gamma or all models are in S.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
        seed_set: Optional[Set[int]] = None,
    ):
        self.coverage_fn = coverage_fn
        self.gamma = float(tolerance_gamma)
        self.seed_set = set(seed_set or set())

        bad = [j for j in self.seed_set if j < 0 or j >= self.coverage_fn.N]
        if bad:
            raise ValueError(
                f"seed_set contains invalid model indices {bad}; expected [0, {self.coverage_fn.N - 1}]"
            )

    def run(self, debug: bool = False) -> PruningResult:
        all_models = set(range(self.coverage_fn.N))
        S: Set[int] = set(self.seed_set)
        history: list[PruningStep] = []
        it = 0

        while True:
            E_now, certs_now = self.coverage_fn.compute_coverage(S, return_certificates=True)
            sum_u = self.coverage_fn.compute_sum_uniqueness(S)

            if E_now <= self.gamma or len(S) == self.coverage_fn.N:
                it += 1
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

            assert certs_now is not None
            remaining = sorted(all_models - S)
            if not remaining:
                # Defensive (should be covered by len(S) == N check above).
                it += 1
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
                return PruningResult(
                    kept_set=set(S),
                    coverage=E_now,
                    sum_uniqueness=sum_u,
                    history=history,
                    certificates=certs_now,
                )

            # Add current bottleneck among not-yet-kept models.
            worst_j = max(remaining, key=lambda j: certs_now[j].uniqueness)
            S.add(worst_j)

            E_after, _ = self.coverage_fn.compute_coverage(S)
            sum_u_after = self.coverage_fn.compute_sum_uniqueness(S)
            it += 1
            history.append(
                PruningStep(
                    iteration=it,
                    removed_model_idx=worst_j,       # here it means "added"
                    removed_model_name=self.coverage_fn.model_names[worst_j],
                    kept_set=set(S),
                    coverage=E_after,
                    sum_uniqueness=sum_u_after,
                    action="add",
                )
            )

            if debug:
                print(
                    f"[warm-forward iter {it}] ADD {self.coverage_fn.model_names[worst_j]} "
                    f"(U={certs_now[worst_j].uniqueness:.6f}) -> E(S)={E_after:.6f}"
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
    """Lazy greedy forward + backward cleanup, with optional k-swap escape.

    Phase 1 (forward / add): Max-heap with lazy re-evaluation.  Greedily add
    the highest-gain (= least-bad) model until E(S) <= gamma.  NOTE: because E is
    a max of residuals (not submodular), the lazy heap gives no speedup -- every
    add invalidates it -- so this computes the same set as plain forward
    selection, modulo gain-tie ordering.  Kept as-is to preserve verified results.

    Phase 2 (backward / prune): Greedy sweep — remove any model from S whose
    removal keeps E(S) <= gamma.

    Phase 3 (k-swap, optional): Escape the Phase 2 single-deletion optimum with
    remove-k / add-(k-1) reduction moves (|S| -> |S|-1 while keeping E <= gamma) --
    the minimal move that monotonicity forbids plain backward from making.
    Optional pure swaps (off by default) are a slow heuristic and rarely help; for
    the strongest result, seed from backward instead (see BackwardKSwapPruner).
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
        max_swap_k: int = 2,
        improvement_mode: str = "first",
        max_candidates: Optional[int] = None,
        allow_pure_swaps: bool = False,
        swap_rel_tol: float = 1e-9,
    ):
        if improvement_mode not in ("first", "best"):
            raise ValueError(f"improvement_mode must be 'first' or 'best', got {improvement_mode!r}")
        self.coverage_fn = coverage_fn
        self.gamma = float(tolerance_gamma)
        self.max_swap_k = int(max_swap_k)
        self.improvement_mode = improvement_mode
        self.max_candidates = max_candidates
        self.allow_pure_swaps = allow_pure_swaps
        # Pure swaps keep |S| fixed, so they are accepted only when they lower E
        # by at least swap_rel_tol * max(|E|, 1).  This margin makes the
        # lexicographic potential strictly decrease -> guaranteed termination.
        self.swap_rel_tol = float(swap_rel_tol)

    def run(self, debug: bool = False) -> PruningResult:
        all_models = set(range(self.coverage_fn.N))
        S: Set[int] = set()
        history: List[PruningStep] = []
        it = 0

        # --- Phase 1: Lazy greedy forward selection ---
        # Bootstrap: pick the first model as the one with lowest E({m}),
        # since E(empty) = inf makes gain-based ranking meaningless.
        best_first_m = -1
        best_first_E = float("inf")
        for m in sorted(all_models):
            E_m, _ = self.coverage_fn.compute_coverage({m})
            if E_m < best_first_E:
                best_first_E = E_m
                best_first_m = m

        it += 1
        S.add(best_first_m)
        E_current = best_first_E
        model_name = self.coverage_fn.model_names[best_first_m]
        sum_u = self.coverage_fn.compute_sum_uniqueness(S)
        history.append(
            PruningStep(
                iteration=it,
                removed_model_idx=best_first_m,
                removed_model_name=model_name,
                kept_set=set(S),
                coverage=E_current,
                sum_uniqueness=sum_u,
                action="add",
            )
        )

        if debug:
            print(f"[Phase 1] Bootstrap: ADD {model_name}  E({{m}})={E_current:.6f}")

        # Generation counter: incremented each time S changes.
        # Heap entries with generation < current are stale.
        generation = 0

        if E_current > self.gamma:
            # Compute initial gains against current S (singleton) and build max-heap.
            # Heap entries: (-gain, timestamp, model_idx, entry_generation)
            # Negate gain because heapq is a min-heap.
            heap: List[tuple] = []
            timestamp = 0
            for m in sorted(all_models - S):
                E_with_m, _ = self.coverage_fn.compute_coverage(S | {m})
                gain = E_current - E_with_m
                heapq.heappush(heap, (-gain, timestamp, m, generation))
                timestamp += 1

            if debug:
                print(f"[Phase 1] E(S)={E_current:.6f}, heap size = {len(heap)}")
                for neg_g, _, m, _ in sorted(heap):
                    name = self.coverage_fn.model_names[m]
                    print(f"  g({name}) = {-neg_g:.6f}")
        else:
            heap = []
            if debug:
                print(f"[Phase 1] E(S)={E_current:.6f} <= gamma={self.gamma} -> DONE after bootstrap")

        phase1_done = False
        while heap and not phase1_done:
            neg_gain, _, m_idx, entry_gen = heapq.heappop(heap)
            gain = -neg_gain

            if m_idx in S:
                # Already added in a previous iteration, skip
                continue

            if entry_gen == generation:
                # Fresh entry.  Phase 1 keeps adding while E(S) > gamma even
                # when the best gain is negative: DISCO weights are fit on
                # Y_fit but U(i|S) is evaluated on Y_eval, so a single add
                # can transiently raise E(S).  Picking the highest-gain
                # (= least-bad) model mirrors regular forward selection,
                # which eventually drives E(S) below gamma.
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
                # Stale entry — recompute gain with current S
                E_with_m, _ = self.coverage_fn.compute_coverage(S | {m_idx})
                new_gain = E_current - E_with_m
                heapq.heappush(heap, (-new_gain, timestamp, m_idx, generation))
                timestamp += 1

        # --- Phase 2: Simple backward pruning ---
        if debug:
            print(f"\n[Phase 2] Starting backward cleanup, |S|={len(S)}, E(S)={E_current:.6f}")

        S, E_current, it = self._backward_cleanup(
            S, it, history=history, debug=debug, phase_label="Phase 2"
        )

        # --- Phase 3: k-swap local search (remove-k-add-(k-1) compound moves) ---
        if self.max_swap_k > 0:
            if debug:
                print(
                    f"\n[Phase 3] Starting k-swap, |S|={len(S)}, E(S)={E_current:.6f}, "
                    f"max_k={self.max_swap_k}, mode={self.improvement_mode}"
                )
            S, E_current, it = self._kswap(
                S, E_current, it, all_models, history=history, debug=debug
            )

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

    def _backward_cleanup(
        self,
        S_in: Set[int],
        it_start: int,
        history: Optional[List[PruningStep]] = None,
        debug: bool = False,
        phase_label: str = "cleanup",
    ) -> Tuple[Set[int], float, int]:
        """Greedy backward sweep: remove any model whose removal keeps E(S) <= gamma.

        If E(S_in) > gamma, no removal is feasible and S is returned unchanged.
        When `history` is provided, append a "remove" step per accepted removal.
        Returns (S, E(S), it).
        """
        S = set(S_in)
        E_cur, _ = self.coverage_fn.compute_coverage(S)
        it = it_start

        changed = True
        while changed:
            changed = False
            for m_idx in sorted(S):
                S_without = S - {m_idx}
                E_without, _ = self.coverage_fn.compute_coverage(S_without)
                if E_without <= self.gamma:
                    S = S_without
                    E_cur = E_without
                    if history is not None:
                        it += 1
                        name = self.coverage_fn.model_names[m_idx]
                        sum_u = self.coverage_fn.compute_sum_uniqueness(S)
                        history.append(
                            PruningStep(
                                iteration=it,
                                removed_model_idx=m_idx,
                                removed_model_name=name,
                                kept_set=set(S),
                                coverage=E_cur,
                                sum_uniqueness=sum_u,
                                action="remove",
                            )
                        )
                        if debug:
                            print(
                                f"[{phase_label}] iter {it}: REMOVE {name}  "
                                f"E(S)={E_cur:.6f}  |S|={len(S)}"
                            )
                    changed = True
                    break  # restart with updated S
        return S, E_cur, it

    def _kswap(
        self,
        S_in: Set[int],
        E_in: float,
        it_start: int,
        all_models: Set[int],
        history: List[PruningStep],
        debug: bool = False,
    ) -> Tuple[Set[int], float, int]:
        """Remove-k-add-(k-1) local search to shrink |S| past the phase-2 local minimum.

        Two move families are tried for each k in 1..max_swap_k:
          - Reduction (n_remove=k, n_add=k-1): removes k models from S and adds k-1
            models from J\\S, reducing |S| by 1.
          - Pure swap (n_remove=k, n_add=k, optional): replaces k models with k
            different ones, keeping |S| fixed.  Accepted only when it strictly
            lowers E by a margin -- the 2nd coordinate of the lexicographic
            potential Phi(S) = (|S|, E(S)).  A pure swap followed by a reduction
            composes into a deeper reduction, so it is only a cheaper, incomplete
            surrogate for a larger max_swap_k; it never makes |S| progress itself.

        Acceptance uses Phi(S) = (|S|, E(S)) compared lexicographically: a
        reduction lowers the first coordinate (any feasible candidate is taken);
        a pure swap must lower the second.  Every accepted move strictly
        decreases Phi over a finite lattice, so the search always terminates,
        |S| never increases, and feasibility (E <= gamma) is preserved.

        Time complexity per outer iteration:
          Let s = |S|, M = |J \\ S| = N - s.
            Reduction neighbourhood:  C(s, k) * C(M, k-1) candidates per k
            Pure-swap neighbourhood:  C(s, k) * C(M, k)   candidates per k
          Each candidate requires one E_hat evaluation costing O(N) QP solves.
          Total per outer while-iteration:
            O(sum_{k=1}^{max_k} [C(s,k)*C(M,k-1) + C(s,k)*C(M,k)] * N_qp)
          where N_qp is the per-evaluation QP cost.

        After every accepted move the search restarts from k=1 so that newly
        opened single-step opportunities are not missed.

        Accepted moves are emitted at INFO level via the module logger so that
        the full search trajectory is auditable without requiring debug=True.
        """
        S = set(S_in)
        E_cur = E_in
        it = it_start

        global_improved = True
        while global_improved:
            global_improved = False

            for k in range(1, self.max_swap_k + 1):
                # Reduction first; pure swap only when no reduction is found.
                move_types: List[Tuple[int, int]] = [(k, k - 1)]
                if self.allow_pure_swaps:
                    move_types.append((k, k))

                for n_remove, n_add in move_types:
                    # Reduction shrinks |S| (1st lexicographic coord); pure swap
                    # keeps it and must instead lower E (2nd coord) by a margin.
                    is_reduction = n_add < n_remove
                    pure_tol = self.swap_rel_tol * max(abs(E_cur), 1.0)

                    sorted_S = sorted(S)
                    available = sorted(all_models - S)  # disjoint from S by construction

                    if len(sorted_S) < n_remove or len(available) < n_add:
                        continue

                    candidates_checked = 0
                    best_move: Optional[Tuple[tuple, tuple, Set[int], float]] = None
                    done = False  # break out of both inner loops

                    for remove_combo in combinations(sorted_S, n_remove):
                        if done:
                            break
                        S_partial = S - set(remove_combo)
                        # remove_combo ⊆ S and available = all_models - S, so they
                        # are disjoint: add_combo never overlaps remove_combo.
                        add_iter = (
                            [()]
                            if n_add == 0
                            else combinations(available, n_add)
                        )

                        for add_combo in add_iter:
                            S_candidate = S_partial | set(add_combo)
                            E_candidate, _ = self.coverage_fn.compute_coverage(S_candidate)
                            candidates_checked += 1

                            if is_reduction:
                                # |S| drops by 1: any feasible candidate lowers Phi.
                                accept = E_candidate <= self.gamma
                            else:
                                # Pure swap keeps |S|: require strict, margin-beating
                                # E improvement (feasibility E < E_cur <= gamma implied).
                                accept = E_candidate < E_cur - pure_tol

                            if accept:
                                if self.improvement_mode == "first":
                                    best_move = (
                                        remove_combo,
                                        tuple(add_combo),
                                        S_candidate,
                                        E_candidate,
                                    )
                                    done = True
                                    break
                                elif best_move is None or E_candidate < best_move[3]:
                                    best_move = (
                                        remove_combo,
                                        tuple(add_combo),
                                        S_candidate,
                                        E_candidate,
                                    )

                            if (
                                self.max_candidates is not None
                                and candidates_checked >= self.max_candidates
                            ):
                                done = True
                                break

                    if best_move is not None:
                        remove_combo, add_combo, S_new, E_new = best_move
                        S_from = set(S)
                        S = S_new
                        E_cur = E_new

                        logger.info(
                            "[kswap k=%d %d-out-%d-in] from=%s → to=%s  "
                            "E_hat=%.6f  |S| %d → %d",
                            k,
                            n_remove,
                            n_add,
                            sorted(S_from),
                            sorted(S),
                            E_cur,
                            len(S_from),
                            len(S),
                        )
                        if debug:
                            rm_names = [self.coverage_fn.model_names[m] for m in remove_combo]
                            ad_names = [self.coverage_fn.model_names[m] for m in add_combo]
                            print(
                                f"[Phase 3 k={k} {n_remove}-out-{n_add}-in]"
                                f"  REMOVE={rm_names}  ADD={ad_names}"
                                f"  from={sorted(S_from)} → to={sorted(S)}"
                                f"  E(S')={E_cur:.6f}  |S'|={len(S)}"
                            )

                        # Record the compound move in history as individual steps.
                        # Additions are listed before removals so the kept_set
                        # snapshot is always feasible (E ≤ γ) for both sub-steps.
                        for m_add in add_combo:
                            it += 1
                            history.append(
                                PruningStep(
                                    iteration=it,
                                    removed_model_idx=m_add,
                                    removed_model_name=self.coverage_fn.model_names[m_add],
                                    kept_set=set(S),
                                    coverage=E_cur,
                                    sum_uniqueness=self.coverage_fn.compute_sum_uniqueness(S),
                                    action="add",
                                )
                            )
                        for m_rem in remove_combo:
                            it += 1
                            history.append(
                                PruningStep(
                                    iteration=it,
                                    removed_model_idx=m_rem,
                                    removed_model_name=self.coverage_fn.model_names[m_rem],
                                    kept_set=set(S),
                                    coverage=E_cur,
                                    sum_uniqueness=self.coverage_fn.compute_sum_uniqueness(S),
                                    action="remove",
                                )
                            )

                        global_improved = True
                        break  # restart from k=1

                if global_improved:
                    break  # restart k loop

        return S, E_cur, it


class BackwardKSwapPruner(PriorityQueuePruner):
    """Backward elimination seed + the k-swap reduction escape.

    Backward elimination halts at a single-deletion local optimum S0: feasible
    (E(S0) <= gamma) but with no single removal staying feasible.  By
    monotonicity of E you cannot remove >= 2 models without adding >= 1 back
    (E(S \\ {a, b}) >= E(S \\ {a}) > gamma), so the minimal escape from S0 is a
    remove-k / add-(k-1) reduction -- exactly what plain backward cannot do.

    Applying that escape from the *backward* seed is sound and dominant:
    reductions only lower |S| while preserving E <= gamma, so the result is
    feasible with ``|result| <= |backward|``, and the search terminates.  Pure
    swaps are inherited but default OFF here, because reduction-only is the
    provably-correct, terminating core; turn them on only to probe plateaus.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
        max_swap_k: int = 2,
        improvement_mode: str = "first",
        max_candidates: Optional[int] = None,
        allow_pure_swaps: bool = False,
        swap_rel_tol: float = 1e-9,
    ):
        super().__init__(
            coverage_fn,
            tolerance_gamma,
            max_swap_k=max_swap_k,
            improvement_mode=improvement_mode,
            max_candidates=max_candidates,
            allow_pure_swaps=allow_pure_swaps,
            swap_rel_tol=swap_rel_tol,
        )

    def run(self, debug: bool = False) -> PruningResult:
        all_models = set(range(self.coverage_fn.N))

        # Phase A: backward elimination seed (a single-deletion local optimum).
        seed = BackwardEliminationPruner(self.coverage_fn, self.gamma).run(debug=debug)
        S: Set[int] = set(seed.kept_set)
        # Carry over backward's trajectory minus its trailing "stop" sentinel so
        # the combined history is continuous.
        history: List[PruningStep] = [s for s in seed.history if s.action != "stop"]
        it = history[-1].iteration if history else 0
        E_cur, _ = self.coverage_fn.compute_coverage(S)

        if debug:
            print(f"\n[BackwardKSwap] seed |S|={len(S)}  E(S)={E_cur:.6f}")

        # Phase B: k-swap reduction escape from the backward seed.
        if self.max_swap_k > 0:
            S, E_cur, it = self._kswap(
                S, E_cur, it, all_models, history=history, debug=debug
            )

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


class ForwardKSwapPruner(PriorityQueuePruner):
    """Forward selection seed + the k-swap reduction escape (no backward cleanup).

    Mirrors :class:`BackwardKSwapPruner` but seeds from forward greedy selection
    instead of backward elimination, then applies the same reduction-only
    (remove-k / add-(k-1)) escape.  The k=1 reduction sub-move already performs
    single-removal cleanup, so this also subsumes a plain backward sweep on the
    forward set.  Pure swaps default OFF.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
        max_swap_k: int = 2,
        improvement_mode: str = "first",
        max_candidates: Optional[int] = None,
        allow_pure_swaps: bool = False,
        swap_rel_tol: float = 1e-9,
    ):
        super().__init__(
            coverage_fn,
            tolerance_gamma,
            max_swap_k=max_swap_k,
            improvement_mode=improvement_mode,
            max_candidates=max_candidates,
            allow_pure_swaps=allow_pure_swaps,
            swap_rel_tol=swap_rel_tol,
        )

    def run(self, debug: bool = False) -> PruningResult:
        all_models = set(range(self.coverage_fn.N))

        # Phase A: forward selection seed.
        seed = ForwardSelectionPruner(self.coverage_fn, self.gamma).run(debug=debug)
        S: Set[int] = set(seed.kept_set)
        history: List[PruningStep] = [s for s in seed.history if s.action != "stop"]
        it = history[-1].iteration if history else 0
        E_cur, _ = self.coverage_fn.compute_coverage(S)

        if debug:
            print(f"\n[ForwardKSwap] seed |S|={len(S)}  E(S)={E_cur:.6f}")

        # Phase B: k-swap reduction escape from the forward seed.
        if self.max_swap_k > 0:
            S, E_cur, it = self._kswap(
                S, E_cur, it, all_models, history=history, debug=debug
            )

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
