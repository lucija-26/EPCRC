"""Exact minimal-representative-set optimum vs greedy (Open Problem 2).

For each random model subset (seed) and each tolerance gamma, compute the TRUE
minimum |S| such that E(S) = max_i U(i|S) <= gamma, by exhaustive search over
subset cardinalities.  Compare against backward / forward / main two-phase
pruner so we can report the optimality gap of each greedy strategy.

Why this is tractable on N<=15:
  - E(S) is gamma-independent, so every U(i|S) is reused across all gammas.
  - A set is infeasible the moment ONE model has U(i|S) > gamma, so we early-exit
    instead of computing all N projections (kills the vast majority of sets fast).
  - Backward elimination gives a feasible set of size `ub`, so the optimum is in
    [1, ub]; we enumerate cardinalities k = 1, 2, ... and stop at the FIRST k that
    admits a feasible set -- that k is provably the exact optimum.

Run from project root:
    python experiments/experiment_exact_optimum.py [n_seeds] [gammas_csv] [subset_size]
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.geometry import DISCOSolver
from epcrc.pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUNDLE = os.path.join(_root, "data", "exp_0_utd19_cache", "bundle.npz")
OUT_DIR = os.path.join(_root, "results", "exact_optimum")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GAMMAS = (
    [float(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
)
SUBSET_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 15
METRIC = "mean_abs"

# Greedy strategies whose optimality gap we report against the exact optimum.
GREEDY_CONFIGS = [
    ("backward", lambda cov, g: BackwardEliminationPruner(cov, g)),
    ("forward",  lambda cov, g: ForwardSelectionPruner(cov, g)),
    # "main" = forward build + backward cleanup (PriorityQueuePruner, no k-swap).
    ("main",     lambda cov, g: PriorityQueuePruner(cov, g, max_swap_k=0)),
    ("backward_kswap2", lambda cov, g: BackwardKSwapPruner(cov, g, max_swap_k=2)),
    ("backward_kswap3", lambda cov, g: BackwardKSwapPruner(cov, g, max_swap_k=3)),
]


class ExactSearcher:
    """Exhaustive minimum-|S| search with an E(S) cache shared across gammas.

    The cache stores, per subset, either the exact E(S) (when fully evaluated on
    a feasible set) or a lower bound on E(S) (when an early-exit proved E > gamma
    for some gamma).  Processing gammas in DESCENDING order makes every cached
    lower bound conclusive for all smaller gammas, so each distinct subset is
    fully evaluated at most once across the whole gamma sweep.
    """

    def __init__(self, Y_fit: np.ndarray, Y_eval: np.ndarray, metric: str):
        self.Y_fit = Y_fit
        self.Y_eval = Y_eval
        self.metric = metric
        self.N = Y_fit.shape[1]
        # frozenset(S) -> ("exact", E) | ("lb", lower_bound_on_E)
        self._cache: Dict[FrozenSet[int], Tuple[str, float]] = {}
        self.n_eval = 0  # number of subsets actually evaluated (cache misses)

    def _evaluate(self, S_list: Tuple[int, ...], gamma: float) -> Tuple[bool, Optional[float]]:
        """Return (feasible, E_if_feasible_else_None) for kept set S at tolerance gamma."""
        key = frozenset(S_list)
        cached = self._cache.get(key)
        if cached is not None:
            kind, val = cached
            if kind == "exact":
                return val <= gamma, (val if val <= gamma else None)
            # kind == "lb": E > val.  Conclusive only when val >= gamma.
            if val >= gamma:
                return False, None
            # Inconclusive lower bound (does not happen in descending-gamma order);
            # fall through and recompute exactly.

        self.n_eval += 1
        S_set = set(S_list)
        Pf = self.Y_fit[:, S_list]
        Pe = self.Y_eval[:, S_list]
        E_max = 0.0
        for i in range(self.N):
            if i in S_set:
                continue
            _, w = DISCOSolver.solve_weights_and_distance(self.Y_fit[:, i], Pf)
            u = DISCOSolver.compute_uniqueness(self.Y_eval[:, i], Pe, w, metric=self.metric)
            if u > gamma:
                # E(S) >= u > gamma -> infeasible; cache the lower bound.
                self._cache[key] = ("lb", u)
                return False, None
            if u > E_max:
                E_max = u
        self._cache[key] = ("exact", E_max)
        return True, E_max

    def min_feasible_size(
        self, gamma: float, ub: int
    ) -> Tuple[int, Tuple[int, ...], float]:
        """Smallest k with a feasible size-k subset, searching k = 1 .. ub.

        `ub` is a known-feasible upper bound (e.g. backward's |S|).  Returns
        (opt_size, opt_set, opt_E).  If no feasible set exists for k < ub, the
        optimum is exactly `ub` (the backward set witnesses feasibility there).
        """
        for k in range(1, ub):
            for combo in combinations(range(self.N), k):
                feasible, E = self._evaluate(combo, gamma)
                if feasible:
                    assert E is not None
                    return k, combo, E
        return ub, tuple(), float("nan")  # opt_set/E filled in by caller from backward


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(BUNDLE, allow_pickle=True)
    Y_fit, Y_eval = d["Y_fit"], d["Y_eval"]
    model_names = [str(x) for x in d["model_names"]]
    N = len(model_names)
    if SUBSET_SIZE > N:
        raise ValueError(f"SUBSET_SIZE={SUBSET_SIZE} > N={N}")

    gammas_desc = sorted(GAMMAS, reverse=True)
    print(f"N={N}  subset_size={SUBSET_SIZE}  seeds={N_SEEDS}  gammas={GAMMAS}")
    print(f"greedy_configs={[c[0] for c in GREEDY_CONFIGS]}\n")

    records: List[dict] = []
    t_start = time.time()
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        sub = sorted(int(i) for i in rng.choice(N, size=SUBSET_SIZE, replace=False))
        sub_names = [model_names[i] for i in sub]
        Yf, Ye = Y_fit[:, sub], Y_eval[:, sub]

        cov = CoverageFunctional(Yf, Ye, sub_names, metric=METRIC)
        searcher = ExactSearcher(Yf, Ye, METRIC)

        t_seed = time.time()
        # Process gammas descending so the E(S) cache is shared maximally.
        for g in gammas_desc:
            # Greedy strategies (cheap) -- also gives the upper bound for exact search.
            greedy_sizes: Dict[str, int] = {}
            greedy_sets: Dict[str, List[str]] = {}
            for cname, factory in GREEDY_CONFIGS:
                res = factory(cov, g).run()
                greedy_sizes[cname] = len(res.kept_set)
                greedy_sets[cname] = [sub_names[i] for i in sorted(res.kept_set)]
            # Every greedy output is feasible by construction, so the smallest
            # one is the tightest upper bound for the exact search.
            ub_name = min(greedy_sizes, key=lambda c: greedy_sizes[c])
            ub = greedy_sizes[ub_name]

            t0 = time.time()
            opt_size, opt_combo, opt_E = searcher.min_feasible_size(g, ub)
            dt = time.time() - t0
            if opt_size == ub and not opt_combo:
                # No feasible set below ub -> the best greedy set IS the optimum.
                opt_set_names = greedy_sets[ub_name]
                opt_E = float(cov.compute_coverage(
                    {i for i in range(SUBSET_SIZE)
                     if sub_names[i] in greedy_sets[ub_name]})[0])
            else:
                opt_set_names = [sub_names[i] for i in opt_combo]

            records.append({
                "seed": seed,
                "subset_global_idx": sub,
                "gamma": g,
                "opt_size": opt_size,
                "opt_E": float(opt_E),
                "opt_set_names": opt_set_names,
                "greedy_sizes": greedy_sizes,
                "gap_vs_opt": {c: greedy_sizes[c] - opt_size for c in greedy_sizes},
                "search_time_s": round(dt, 3),
            })
        print(
            f"  seed {seed}: subset={sub_names}  "
            f"evals={searcher.n_eval}  ({time.time()-t_seed:.1f}s)"
        )

    # ---- aggregate: per gamma, mean optimum & mean greedy sizes / gaps ----
    summary: Dict[str, dict] = {}
    for g in sorted(GAMMAS):
        rs = [r for r in records if r["gamma"] == g]
        summary[str(g)] = {
            "mean_opt_size": float(np.mean([r["opt_size"] for r in rs])),
            "mean_greedy_size": {
                c: float(np.mean([r["greedy_sizes"][c] for r in rs]))
                for c, _ in GREEDY_CONFIGS
            },
            "mean_gap_vs_opt": {
                c: float(np.mean([r["gap_vs_opt"][c] for r in rs]))
                for c, _ in GREEDY_CONFIGS
            },
            "n_seeds": len(rs),
        }

    payload = {
        "experiment": "exact_optimum",
        "config": {
            "n_models_total": N,
            "subset_size": SUBSET_SIZE,
            "n_seeds": N_SEEDS,
            "gammas": sorted(GAMMAS),
            "metric": METRIC,
            "greedy_configs": [c[0] for c in GREEDY_CONFIGS],
        },
        "model_names_full": model_names,
        "summary": summary,
        "records": records,
    }
    out = os.path.join(OUT_DIR, f"exact_optimum_{N_SEEDS}seeds.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[io] wrote {out}  ({len(records)} records, {time.time()-t_start:.1f}s total)")

    # ---- console table: mean size (and gap) per gamma ----
    print("\nmean |S| over seeds   [gap vs optimum in brackets]:")
    cfg_names = [c[0] for c in GREEDY_CONFIGS]
    hdr = f"{'gamma':>6} {'OPT':>6} " + " ".join(f"{c:>20}" for c in cfg_names)
    print(hdr)
    for g in sorted(GAMMAS):
        s = summary[str(g)]
        cells = [
            f"{s['mean_greedy_size'][c]:5.1f} [+{s['mean_gap_vs_opt'][c]:.1f}]"
            for c in cfg_names
        ]
        print(f"{g:6.0f} {s['mean_opt_size']:6.1f} " + " ".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
