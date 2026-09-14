"""Experiment E0 -- individual redundancy certificates do not compose (claim C1).

For each tolerance gamma:

  1. measure every judge's leave-one-out error U(i | J \\ {i});
  2. collect the individually removable set R = {i : U(i | J \\ {i}) <= gamma};
  3. delete all of R at once, giving the naive retained set S_naive = J \\ R;
  4. evaluate the joint coverage of S_naive;
  5. compare against the minimum jointly feasible panel;
  6. read a dependency graph off the leave-one-out weights and look for cycles.

The composition gap is (minimum feasible size) - |S_naive|: how many judges the
one-at-a-time audit claimed could go but the joint constraint says must stay.

What the claim actually rests on is step 4: if S_naive breaches gamma then the
individual certificates demonstrably do not compose, and that is one coverage
evaluation.  Step 5 quantifies the damage and is the expensive half, because the
minimum is found by enumerating subsets in increasing size.  That is exact and
cheap for the controlled constructions but hopeless at N = 19, so it is capped by
`EXHAUSTIVE_EVAL_BUDGET`; past the cap the size is reported as an interval
(`min_feasible_lower_bound` certified by the layers that did finish, upper bound
from backward elimination) and `min_feasible_is_exact` is False.  The sign of a
bounded composition gap proves nothing and must not be read as evidence.

Run on the two controlled constructions whose answer is known in advance, over
several seeds, and write results/e0_noncomposability.json.

`--real` repeats the same analysis on a scored panel over the five grouped split
seeds (plan section 22 step 8) and writes results/e0_real_<panel>.json.  Only the
FIT/CERT partition changes between those seeds, so no additional GPU inference is
involved.

Usage:

    python -u experiments/experiment_e0_noncomposability.py
    python -u experiments/experiment_e0_noncomposability.py --real
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from math import comb
from typing import Dict, List, NamedTuple, Set, Tuple

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
# Coverage evaluations the exhaustive minimum-panel search may spend per instance.
# Our engineering choice, not the plan's -- section 22 fixes no such budget.  The
# whole power set of the controlled constructions (N = 8, 9) fits well inside it,
# so they stay exact; the real panel does not come close and falls back to a
# bounded answer.
#
# Raising it would not buy exactness on the real panel.  At N = 20 the minimum
# sits near 15-18 judges, and certifying that by enumeration means clearing every
# smaller layer first -- on the order of 700k subsets, C(20, 15) = 15504 in the
# size-15 layer alone, per split seed.  So C1 is built on naive_violates_gamma,
# which is exact everywhere, and C6 is specified on subpanels of size 12-16,
# where the power set is small enough to enumerate outright.
#
# Evaluations are memoised per panel, so the cost is paid once per split seed
# rather than once per tolerance.
EXHAUSTIVE_EVAL_BUDGET = 20000

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


class MinFeasible(NamedTuple):
    """Smallest panel meeting a tolerance, with how well that size is pinned down.

    `size` is exact when `is_exact`, and otherwise only an upper bound produced by
    backward elimination.  `lower_bound` is always certified: it is one more than
    the largest subset size that was enumerated in full without finding anything
    feasible, so no panel smaller than that exists.  When the two coincide the
    answer is exact even though enumeration was cut short.
    """
    size: int
    subset: List[int]
    error: float
    is_exact: bool
    lower_bound: int
    method: str


def greedy_feasible_panel(
    cov: JudgeCoverageFunctional, gamma: float
) -> Tuple[int, List[int], float]:
    """Backward elimination down to the tolerance: an upper bound on the minimum.

    The full panel reconstructs itself at zero error and so is feasible for every
    tolerance; judges are then dropped one at a time, always the one whose removal
    leaves the smallest coverage error, and the walk stops when the next step
    would breach gamma.  Costs O(N^2) coverage evaluations against the
    combinatorially many an exhaustive search needs.
    """
    current = set(range(cov.N))
    while len(current) > 1:
        error, drop, trial = min(
            (
                (cov.compute_coverage(current - {j})[0], j, current - {j})
                for j in sorted(current)
            ),
            key=lambda candidate: candidate[0],
        )
        if error > gamma:
            break
        current = trial

    kept = sorted(current)
    error, _ = cov.compute_coverage(set(kept))
    return len(kept), kept, float(error)


def min_feasible_panel(
    cov: JudgeCoverageFunctional,
    gamma: float,
    max_evals: int = EXHAUSTIVE_EVAL_BUDGET,
) -> MinFeasible:
    """Smallest panel meeting the tolerance, exhaustively while that is affordable.

    Enumerating subsets in increasing size removes any doubt about greedy
    artefacts, which is why the controlled instances are done that way: at N = 8
    or 9 the whole power set is a few hundred coverage evaluations.

    It does not survive the real panel.  At N = 19 the size-10 layer alone is
    92378 subsets, and every tolerance and split seed would repeat it, so the
    search is capped at `max_evals` coverage evaluations.  Past the cap the
    function stops enumerating and reports backward elimination's answer as an
    upper bound, together with the lower bound the completed layers certify.
    Callers must consult `is_exact` before treating the size as the true minimum,
    because a bound alone cannot establish the sign of the composition gap.
    """
    N = cov.N
    evaluations = 0

    for k in range(1, N + 1):
        layer = comb(N, k)
        if evaluations + layer > max_evals:
            size, subset, error = greedy_feasible_panel(cov, gamma)
            return MinFeasible(
                size=size, subset=subset, error=error,
                is_exact=size == k, lower_bound=k, method="greedy_bound",
            )

        for combo in combinations(range(N), k):
            error, _ = cov.compute_coverage(set(combo))
            evaluations += 1
            if error <= gamma:
                return MinFeasible(
                    size=k, subset=list(combo), error=float(error),
                    is_exact=True, lower_bound=k, method="exhaustive",
                )

    # Unreachable: the full panel reconstructs itself at zero error.
    return MinFeasible(
        size=N, subset=list(range(N)), error=0.0,
        is_exact=True, lower_bound=N, method="exhaustive",
    )


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
    max_evals: int = EXHAUSTIVE_EVAL_BUDGET,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """The section 22 procedure for one panel, over the tolerance grid.

    `verbose` traces each tolerance as it completes.  On the real panel one
    tolerance can spend the whole exhaustive budget, so a silent run of several
    hours is indistinguishable from a hang; the synthetic instances finish in
    seconds and stay quiet.
    """
    N = cov.N

    loo_errors, loo_weights = leave_one_out(cov)
    graph = dependency_cycles(loo_weights, DEPENDENCY_TAU)

    declared = set(gammas)
    if add_loo_breakpoints:
        gammas = sorted(set(gammas) | set(loo_breakpoints(loo_errors)))

    rows: List[Dict[str, object]] = []
    for index, gamma in enumerate(gammas, start=1):
        removable = [i for i in range(N) if loo_errors[i] <= gamma]
        naive = [i for i in range(N) if i not in set(removable)]

        started = time.time()
        naive_error, _ = cov.compute_coverage(set(naive))
        opt = min_feasible_panel(cov, gamma, max_evals)
        if verbose:
            print(f"  [{instance} seed={seed}] gamma {gamma:.3f} "
                  f"({index}/{len(gammas)})  min|S|={opt.size} "
                  f"{'exact' if opt.is_exact else 'bound'}  "
                  f"{time.time() - started:.1f}s", flush=True)

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
            "min_feasible_size": opt.size,
            "min_feasible_set": [names[i] for i in opt.subset],
            "min_feasible_coverage": opt.error,
            "min_feasible_is_exact": opt.is_exact,
            "min_feasible_lower_bound": opt.lower_bound,
            "min_feasible_method": opt.method,
            "composition_gap": opt.size - len(naive),
            "composition_gap_is_exact": opt.is_exact,
            "dependency_graph": graph,
        })

    return rows


def run_instance(name: str, seed: int) -> List[Dict[str, object]]:
    fit, eval_, names = INSTANCES[name](seed)
    cov = JudgeCoverageFunctional(fit, eval_, names)
    return analyse(cov, names, name, seed, GAMMAS)


def run_real(
    split_seed: int,
    panel_name: str = "core8",
    max_evals: int = EXHAUSTIVE_EVAL_BUDGET,
) -> List[Dict[str, object]]:
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
        add_loo_breakpoints=True, max_evals=max_evals, verbose=True,
    )


def summarise(records: List[Dict[str, object]], instances: List[str],
              gammas: List[float]) -> None:
    header = (
        f"{'instance':<22}{'gamma':>8}{'removable':>11}{'S_naive':>9}"
        f"{'E(S_naive)':>12}{'min|S|':>8}{'gap':>6}{'cycles':>8}{'exact':>7}"
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
            # A bounded min|S| makes the gap a bound too, and the sign of a bound
            # proves nothing, so the reader has to be able to see which is which.
            exact = "yes" if all(r["min_feasible_is_exact"] for r in group) else "BOUND"
            print(
                f"{name:<22}{gamma:>8.3f}{removable:>11.1f}{naive_size:>9.1f}"
                f"{naive_cov:>12.4f}{opt:>8.1f}{gap:>6.1f}{cycles:>8.1f}{exact:>7}"
            )


def main_real(
    panel_name: str = "core8",
    out: str = None,
    max_evals: int = EXHAUSTIVE_EVAL_BUDGET,
) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = out or os.path.join(RESULTS, f"e0_real_{panel_name}.json")

    records: List[Dict[str, object]] = []
    for split_seed in SPLIT_SEEDS:
        records.extend(run_real(split_seed, panel_name, max_evals))
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
            "exhaustive_eval_budget": max_evals,
            "min_feasible_all_exact": all(
                r["min_feasible_is_exact"] for r in records
            ),
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

    bounded = [r for r in records if not r["min_feasible_is_exact"]]
    if bounded:
        worst = max(r["min_feasible_size"] - r["min_feasible_lower_bound"]
                    for r in bounded)
        print(
            f"\nNOTE: {len(bounded)} of {len(records)} rows exceeded the "
            f"{max_evals}-evaluation exhaustive budget at N={records[0]['n_judges']}, "
            f"so min|S| there is backward elimination's upper bound, certified no "
            f"lower than min_feasible_lower_bound (widest interval {worst} judges). "
            f"The composition gap is correspondingly an upper bound on those rows "
            f"and its sign must not be read as evidence. What C1 rests on is "
            f"naive_violates_gamma, which is exact everywhere."
        )


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
    parser.add_argument("--max-exhaustive-evals", type=int,
                        default=EXHAUSTIVE_EVAL_BUDGET,
                        help="coverage evaluations the exact minimum-panel search "
                             "may spend before falling back to a bounded answer")
    args = parser.parse_args()
    if args.real:
        main_real(args.panel, args.out, args.max_exhaustive_evals)
    else:
        main()
