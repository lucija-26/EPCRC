"""Experiment E0 -- individual redundancy certificates do not compose (claim C1).

For each tolerance gamma:

  1. measure every judge's leave-one-out error U(i | J \\ {i});
  2. collect the individually removable set R = {i : U(i | J \\ {i}) <= gamma};
  3. delete all of R at once, giving the naive retained set S_naive = J \\ R;
  4. evaluate the joint coverage of S_naive;
  5. compare against the exact minimum jointly feasible panel;
  6. read a dependency graph off the leave-one-out weights and look for cycles.

The composition gap is (minimum feasible size) - |S_naive|: how many judges the
one-at-a-time audit claimed could go but the joint constraint says must stay.

Run on the two controlled constructions whose answer is known in advance, over
several seeds, and write results/e0_noncomposability.json.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeCoverageFunctional
from epcrc.synthetic_judges import curved_arc, duplicated_extremes

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
OUT = os.path.join(RESULTS, "e0_noncomposability.json")

GAMMAS = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
SEEDS = [0, 1, 2, 3, 4]
# A leave-one-out weight above this counts as "i leans on j" in the dependency graph.
DEPENDENCY_TAU = 0.05

INSTANCES = {
    "duplicated_extremes": lambda seed: duplicated_extremes(
        n_items=80, n_contexts=2, seed=seed
    ),
    "curved_arc_9": lambda seed: curved_arc(
        n_judges=9, n_items=80, n_contexts=2, seed=seed
    ),
}


def leave_one_out(cov: JudgeCoverageFunctional) -> Tuple[np.ndarray, np.ndarray]:
    """Per-judge LOO error and the weight matrix of the LOO certificates."""
    N = cov.N
    all_judges = set(range(N))
    errors = np.empty(N, dtype=float)
    weights = np.zeros((N, N), dtype=float)

    for i in range(N):
        peers = sorted(all_judges - {i})
        cert = cov.compute_certificate(i, set(peers))
        errors[i] = cert.uniqueness
        weights[i, peers] = cert.weights

    return errors, weights


def min_feasible_panel(
    cov: JudgeCoverageFunctional, gamma: float
) -> Tuple[int, List[int], float]:
    """Smallest panel meeting the tolerance, by exhaustive search.

    The controlled instances have well under 20 judges, so enumerating subsets
    in increasing size is cheap and removes any doubt about greedy artefacts.
    """
    N = cov.N
    for k in range(1, N + 1):
        for combo in combinations(range(N), k):
            error, _ = cov.compute_coverage(set(combo))
            if error <= gamma:
                return k, list(combo), float(error)
    return N, list(range(N)), 0.0


def dependency_cycles(weights: np.ndarray, tau: float) -> Dict[str, object]:
    """Strongly connected components of the 'i leans on j' graph.

    Judges in a common component reconstruct each other, so their individual
    redundancy certificates are mutually dependent and cannot all be honoured.
    """
    adjacency = weights > tau
    N = adjacency.shape[0]

    reach = adjacency.copy()
    for _ in range(N):
        updated = reach | (reach @ adjacency)
        if np.array_equal(updated, reach):
            break
        reach = updated

    mutual = reach & reach.T
    components: List[List[int]] = []
    seen: Set[int] = set()
    for i in range(N):
        if i in seen:
            continue
        component = sorted(np.flatnonzero(mutual[i]).tolist() + [i])
        component = sorted(set(component))
        seen.update(component)
        if len(component) > 1:
            components.append(component)

    return {
        "n_edges": int(adjacency.sum()),
        "cycles": components,
        "n_cycles": len(components),
        "largest_cycle": max((len(c) for c in components), default=0),
    }


def run_instance(name: str, seed: int) -> List[Dict[str, object]]:
    fit, eval_, names = INSTANCES[name](seed)
    cov = JudgeCoverageFunctional(fit, eval_, names)
    N = cov.N

    loo_errors, loo_weights = leave_one_out(cov)
    graph = dependency_cycles(loo_weights, DEPENDENCY_TAU)

    rows: List[Dict[str, object]] = []
    for gamma in GAMMAS:
        removable = [i for i in range(N) if loo_errors[i] <= gamma]
        naive = [i for i in range(N) if i not in set(removable)]

        naive_error, _ = cov.compute_coverage(set(naive))
        opt_size, opt_set, opt_error = min_feasible_panel(cov, gamma)

        rows.append({
            "instance": name,
            "seed": seed,
            "gamma": gamma,
            "n_judges": N,
            "loo_errors": [float(e) for e in loo_errors],
            "individually_removable": [names[i] for i in removable],
            "n_individually_removable": len(removable),
            "frac_individually_removable": len(removable) / N,
            "naive_retained": [names[i] for i in naive],
            "naive_retained_size": len(naive),
            "naive_coverage": float(naive_error),
            "naive_violates_gamma": bool(naive_error > gamma),
            "naive_violation_amount": float(max(0.0, naive_error - gamma)),
            "min_feasible_size": opt_size,
            "min_feasible_set": [names[i] for i in opt_set],
            "min_feasible_coverage": opt_error,
            "composition_gap": opt_size - len(naive),
            "dependency_graph": graph,
        })

    return rows


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    records: List[Dict[str, object]] = []

    for name in INSTANCES:
        for seed in SEEDS:
            rows = run_instance(name, seed)
            records.extend(rows)
            print(f"[{name} seed={seed}] done", flush=True)

    with open(OUT, "w") as handle:
        json.dump({"gammas": GAMMAS, "seeds": SEEDS, "records": records}, handle, indent=2)

    print(f"\nwrote {OUT}\n")
    header = (
        f"{'instance':<22}{'gamma':>8}{'removable':>11}{'S_naive':>9}"
        f"{'E(S_naive)':>12}{'min|S|':>8}{'gap':>6}{'cycles':>8}"
    )
    print(header)
    print("-" * len(header))

    for name in INSTANCES:
        for gamma in GAMMAS:
            group = [
                r for r in records if r["instance"] == name and r["gamma"] == gamma
            ]
            removable = np.mean([r["n_individually_removable"] for r in group])
            naive_size = np.mean([r["naive_retained_size"] for r in group])
            naive_cov = np.mean([r["naive_coverage"] for r in group])
            opt = np.mean([r["min_feasible_size"] for r in group])
            gap = np.mean([r["composition_gap"] for r in group])
            cycles = np.mean([r["dependency_graph"]["n_cycles"] for r in group])
            print(
                f"{name:<22}{gamma:>8.3f}{removable:>11.1f}{naive_size:>9.1f}"
                f"{naive_cov:>12.4f}{opt:>8.1f}{gap:>6.1f}{cycles:>8.1f}"
            )


if __name__ == "__main__":
    main()
