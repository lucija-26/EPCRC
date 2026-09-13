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

`--real` repeats the same analysis on the scored Core-8 panel over the five
grouped split seeds (plan section 22 step 8) and writes
results/e0_real_core8.json.  Only the FIT/CERT partition changes between those
seeds, so no additional GPU inference is involved.

Usage:

    python -u experiments/experiment_e0_noncomposability.py
    python -u experiments/experiment_e0_noncomposability.py --real
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeCoverageFunctional
from epcrc.panel import PANELS, SPLIT_SEEDS, load_panel, scores_dir
from epcrc.rewardbench import PRIMARY_SEED
from epcrc.synthetic_judges import curved_arc, duplicated_extremes

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
OUT = os.path.join(RESULTS, "e0_noncomposability.json")
REAL_OUT = os.path.join(RESULTS, "e0_real_core8.json")

GAMMAS = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
SEEDS = [0, 1, 2, 3, 4]
# Real judges live at a completely different error scale from the controlled
# constructions, so the real run uses the predeclared E1 grid of section 23.
REAL_GAMMAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
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


def loo_breakpoints(loo_errors: np.ndarray) -> List[float]:
    """The only tolerances at which the individually-removable set changes.

    ``R(gamma) = {i : U_i <= gamma}`` grows by one judge exactly when gamma
    crosses a leave-one-out error, and is constant in between.  So the sorted
    LOO errors are the complete set of informative tolerances: any other grid
    either repeats one of these sets or -- if every LOO error sits above the
    grid, which is what happens on a small real panel -- reports an empty R
    and says nothing about composition at all.

    These are read off the data, so they are reported as a diagnostic beside
    the predeclared grid and never in place of it.
    """
    return sorted({float(np.nextafter(e, np.inf)) for e in loo_errors})


def analyse(
    cov: JudgeCoverageFunctional,
    names: List[str],
    instance: str,
    seed: int,
    gammas: List[float],
    add_loo_breakpoints: bool = False,
) -> List[Dict[str, object]]:
    """The section 22 procedure for one panel, over the tolerance grid."""
    N = cov.N

    loo_errors, loo_weights = leave_one_out(cov)
    graph = dependency_cycles(loo_weights, DEPENDENCY_TAU)

    declared = set(gammas)
    if add_loo_breakpoints:
        gammas = sorted(set(gammas) | set(loo_breakpoints(loo_errors)))

    rows: List[Dict[str, object]] = []
    for gamma in gammas:
        removable = [i for i in range(N) if loo_errors[i] <= gamma]
        naive = [i for i in range(N) if i not in set(removable)]

        naive_error, _ = cov.compute_coverage(set(naive))
        opt_size, opt_set, opt_error = min_feasible_panel(cov, gamma)

        rows.append({
            "instance": instance,
            "seed": seed,
            "gamma": gamma,
            "gamma_source": "declared" if gamma in declared else "loo_breakpoint",
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


def run_instance(name: str, seed: int) -> List[Dict[str, object]]:
    fit, eval_, names = INSTANCES[name](seed)
    cov = JudgeCoverageFunctional(fit, eval_, names)
    return analyse(cov, names, name, seed, GAMMAS)


def run_real(split_seed: int, panel_name: str = "core8") -> List[Dict[str, object]]:
    """Same procedure on a scored panel, fitted on FIT and judged on CERT.

    TEST stays locked for E1, so C1 never consults it.
    """
    panel = load_panel(
        PRIMARY_SEED, scores_dir(panel_name), split_seed=split_seed
    )
    cov = JudgeCoverageFunctional(
        panel.splits["FIT"], panel.splits["CERT"], panel.judge_ids
    )
    return analyse(
        cov, panel.judge_ids, f"{panel_name}_real", split_seed, REAL_GAMMAS,
        add_loo_breakpoints=True,
    )


def summarise(records: List[Dict[str, object]], instances: List[str],
              gammas: List[float]) -> None:
    header = (
        f"{'instance':<22}{'gamma':>8}{'removable':>11}{'S_naive':>9}"
        f"{'E(S_naive)':>12}{'min|S|':>8}{'gap':>6}{'cycles':>8}"
    )
    print(header)
    print("-" * len(header))

    for name in instances:
        for gamma in gammas:
            group = [
                r for r in records if r["instance"] == name and r["gamma"] == gamma
            ]
            if not group:
                continue
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


def main_real(panel_name: str = "core8", out: str = None) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = out or os.path.join(RESULTS, f"e0_real_{panel_name}.json")

    records: List[Dict[str, object]] = []
    for split_seed in SPLIT_SEEDS:
        records.extend(run_real(split_seed, panel_name))
        print(f"[{panel_name}_real split_seed={split_seed}] done", flush=True)

    with open(out, "w") as handle:
        json.dump({
            "experiment": "E0",
            "claim": "C1",
            "panel": panel_name,
            "pairs_seed": PRIMARY_SEED,
            "gammas": REAL_GAMMAS,
            "loo_breakpoints_included": True,
            "split_seeds": SPLIT_SEEDS,
            "records": records,
        }, handle, indent=2)

    print(f"\nwrote {out}\n")
    loo = np.array([r["loo_errors"] for r in records])
    print(f"leave-one-out error over judges and seeds: "
          f"min {loo.min():.4f}  median {np.median(loo):.4f}  max {loo.max():.4f}\n")

    print("predeclared grid (section 23)")
    summarise(records, [f"{panel_name}_real"], REAL_GAMMAS)

    if loo.min() > max(REAL_GAMMAS):
        print(f"\nNOTE: every judge's leave-one-out error exceeds the largest "
              f"predeclared gamma ({max(REAL_GAMMAS)}), so R(gamma) is empty "
              f"throughout that grid and C1 has no bite on it.")

    breaks = sorted({r["gamma"] for r in records
                     if r["gamma_source"] == "loo_breakpoint"})
    if breaks:
        print("\nleave-one-out breakpoints (data-driven diagnostic, per seed)")
        summarise(records, [f"{panel_name}_real"], breaks)


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
    summarise(records, list(INSTANCES), GAMMAS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true",
                        help="run on a scored panel over the five split seeds")
    parser.add_argument("--panel", choices=sorted(PANELS), default="core8",
                        help="which scored panel --real should use")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main_real(args.panel, args.out) if args.real else main()
