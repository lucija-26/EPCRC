"""Risk-controlled pruning (paper Open Problem 1 / section 4.3, Experiment 4).

Point estimates Uhat(i|S) are noisy: a removal that looks feasible can violate
the true coverage constraint.  The certified rule removes j only when an UPPER
CONFIDENCE BOUND on the resulting coverage stays below gamma:

    UCB(i|S) = mean(|R_i|) + z_{delta/N} * std(|R_i|) / sqrt(n)
    E_UCB(S) = max_i UCB(i|S)          remove j only if E_UCB(S \\ {j}) <= gamma

where R_i are the per-row eval residuals of the (fit-sample) certificate and
z_{delta/N} is the normal quantile after a union bound over the N models, so

    P( max_i U(i|S) <= gamma )  >=  1 - delta    whenever  E_UCB(S) <= gamma

holds asymptotically for each FIXED S.  The paper's open problem is exactly
what this module does NOT give: bounds uniform over the adaptive sequence of
sets a pruning run visits (the union bound covers models, not sets).  Honest
framing for the report: this is the practical certified rule the paper
proposes; making it adaptively valid is the open problem.

Only mean_abs uniqueness is supported (the CLT is on the mean of |R_i|).
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np
from scipy.stats import norm

from .coverage import CoverageFunctional
from .pruning import PruningResult, PruningStep


def uniqueness_ucb(
    cov: CoverageFunctional,
    kept_set: Set[int],
    delta: float = 0.05,
) -> Dict[int, float]:
    """Per-model UCB on U(i|S), union-bounded over the N models."""
    if cov.metric != "mean_abs":
        raise ValueError(f"UCB requires metric='mean_abs', got {cov.metric!r}")
    _, certs = cov.compute_coverage(kept_set, return_certificates=True)
    assert certs is not None
    Pe = cov.Y_eval[:, sorted(kept_set)]
    n = Pe.shape[0]
    z = float(norm.ppf(1.0 - delta / cov.N))

    ucb: Dict[int, float] = {}
    for i in range(cov.N):
        if i in kept_set:
            ucb[i] = 0.0
            continue
        r = np.abs(cov.Y_eval[:, i] - Pe @ certs[i].weights)
        ucb[i] = float(r.mean() + z * r.std(ddof=1) / np.sqrt(n))
    return ucb


def coverage_ucb(
    cov: CoverageFunctional,
    kept_set: Set[int],
    delta: float = 0.05,
) -> float:
    """E_UCB(S) = max_i UCB(i|S)."""
    return max(uniqueness_ucb(cov, kept_set, delta).values())


class RiskControlledBackwardPruner:
    """Backward elimination that accepts a removal only when E_UCB <= gamma.

    Strictly more conservative than BackwardEliminationPruner (UCB >= point
    estimate), so it returns a superset-sized kept set whose feasibility is
    certified at level 1 - delta per visited set.
    """

    def __init__(
        self,
        coverage_fn: CoverageFunctional,
        tolerance_gamma: float,
        delta: float = 0.05,
    ):
        self.coverage_fn = coverage_fn
        self.gamma = float(tolerance_gamma)
        self.delta = float(delta)

    def run(self, debug: bool = False) -> PruningResult:
        S = set(range(self.coverage_fn.N))
        history: list[PruningStep] = []
        it = 0

        while True:
            it += 1
            best_j, best_ucb = None, float("inf")
            for j in sorted(S):
                ucb = coverage_ucb(self.coverage_fn, S - {j}, self.delta)
                if ucb <= self.gamma and ucb < best_ucb:
                    best_j, best_ucb = j, ucb

            if best_j is None:
                E_now, certs = self.coverage_fn.compute_coverage(S, return_certificates=True)
                assert certs is not None
                history.append(PruningStep(
                    iteration=it, removed_model_idx=None, removed_model_name=None,
                    kept_set=set(S), coverage=E_now,
                    sum_uniqueness=self.coverage_fn.compute_sum_uniqueness(S),
                    action="stop",
                ))
                return PruningResult(
                    kept_set=set(S), coverage=E_now,
                    sum_uniqueness=self.coverage_fn.compute_sum_uniqueness(S),
                    history=history, certificates=certs,
                )

            S.remove(best_j)
            E_after, _ = self.coverage_fn.compute_coverage(S)
            history.append(PruningStep(
                iteration=it, removed_model_idx=best_j,
                removed_model_name=self.coverage_fn.model_names[best_j],
                kept_set=set(S), coverage=E_after,
                sum_uniqueness=self.coverage_fn.compute_sum_uniqueness(S),
                action="remove",
            ))
            if debug:
                print(
                    f"[risk-bwd iter {it}] REMOVE {self.coverage_fn.model_names[best_j]}"
                    f"  E_UCB={best_ucb:.6f}  E_hat={E_after:.6f}  |S|={len(S)}"
                )
