"""Subset-robustness sweep: compare 6 pruning configs over random model subsets.

Train/load the full ecosystem ONCE (cached bundle), then for each of N_SEEDS
seeds draw a random SUBSET_SIZE-of-N model subset and run all six configs across
the gamma sweep on that sub-ecosystem.  Results (per seed x config x gamma) are
written to JSON for plotting (mean |S| vs gamma with spread over seeds).

Six configs (each is a composition of forward/backward build, backward cleanup,
and the reduction-only k-swap escape):

    backward                 backward elimination
    backward_kswap           backward  -> k-swap escape
    forward                  forward selection
    forward_kswap            forward   -> k-swap escape
    forward_backward         forward   -> backward cleanup            (the 2-phase)
    forward_backward_kswap   forward   -> backward cleanup -> k-swap   (2-phase + escape)

Run from project root:
    python experiments/experiment_subset_robustness.py [n_seeds] [gammas_csv]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUNDLE = os.path.join(_root, "data", "exp_0_utd19_cache", "bundle.npz")
OUT_DIR = os.path.join(_root, "results", "subset_robustness")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
GAMMAS = (
    [float(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
)
SUBSET_SIZE = 15
METRIC = "mean_abs"
KSWAP_MAX_K = 2

# (name, factory(coverage_fn, gamma) -> pruner with .run())
CONFIGS = [
    ("backward",               lambda cov, g: BackwardEliminationPruner(cov, g)),
    ("backward_kswap",         lambda cov, g: BackwardKSwapPruner(cov, g, max_swap_k=KSWAP_MAX_K)),
    ("forward",                lambda cov, g: ForwardSelectionPruner(cov, g)),
    ("forward_kswap",          lambda cov, g: ForwardKSwapPruner(cov, g, max_swap_k=KSWAP_MAX_K)),
    ("forward_backward",       lambda cov, g: PriorityQueuePruner(cov, g, max_swap_k=0)),
    ("forward_backward_kswap", lambda cov, g: PriorityQueuePruner(cov, g, max_swap_k=KSWAP_MAX_K)),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(BUNDLE, allow_pickle=True)
    Y_fit, Y_eval = d["Y_fit"], d["Y_eval"]
    model_names = [str(x) for x in d["model_names"]]
    N = len(model_names)
    if SUBSET_SIZE > N:
        raise ValueError(f"SUBSET_SIZE={SUBSET_SIZE} > N={N}")

    print(f"N={N}  subset_size={SUBSET_SIZE}  seeds={N_SEEDS}  gammas={GAMMAS}")
    print(f"configs={[c[0] for c in CONFIGS]}\n")

    records = []
    t_start = time.time()
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        sub = sorted(int(i) for i in rng.choice(N, size=SUBSET_SIZE, replace=False))
        sub_names = [model_names[i] for i in sub]
        # ONE coverage_fn per subset: E(S) is gamma-independent, so its cache is
        # shared across all gammas and configs for this seed (big speedup).
        cov = CoverageFunctional(Y_fit[:, sub], Y_eval[:, sub], sub_names, metric=METRIC)

        t_seed = time.time()
        for g in GAMMAS:
            for cname, factory in CONFIGS:
                t0 = time.time()
                res = factory(cov, g).run()
                dt = time.time() - t0
                kept_local = sorted(int(i) for i in res.kept_set)
                records.append({
                    "seed": seed,
                    "subset_global_idx": sub,
                    "gamma": g,
                    "config": cname,
                    "kept_size": len(res.kept_set),
                    "coverage_E": float(res.coverage),
                    "feasible": bool(res.coverage <= g),
                    "kept_names": [sub_names[i] for i in kept_local],
                    "wall_time_s": round(dt, 3),
                })
        print(f"  seed {seed}: subset={sub_names}  ({time.time()-t_seed:.1f}s)")

    # ---- aggregate for plotting: per config, per gamma -> stats over seeds ----
    summary = {}
    for cname, _ in CONFIGS:
        summary[cname] = {}
        for g in GAMMAS:
            sizes = [r["kept_size"] for r in records if r["config"] == cname and r["gamma"] == g]
            feas = [r["feasible"] for r in records if r["config"] == cname and r["gamma"] == g]
            summary[cname][str(g)] = {
                "mean_size": float(np.mean(sizes)),
                "std_size": float(np.std(sizes)),
                "min_size": int(np.min(sizes)),
                "max_size": int(np.max(sizes)),
                "feasible_rate": float(np.mean(feas)),
                "n_seeds": len(sizes),
            }

    payload = {
        "experiment": "subset_robustness",
        "config": {
            "n_models_total": N,
            "subset_size": SUBSET_SIZE,
            "n_seeds": N_SEEDS,
            "gammas": GAMMAS,
            "metric": METRIC,
            "kswap_max_k": KSWAP_MAX_K,
            "configs": [c[0] for c in CONFIGS],
        },
        "model_names_full": model_names,
        "summary": summary,
        "records": records,
    }
    out = os.path.join(OUT_DIR, f"subset_robustness_{N_SEEDS}seeds.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[io] wrote {out}  ({len(records)} records, {time.time()-t_start:.1f}s total)")

    # ---- console table: mean |S| (std) per config x gamma ----
    print("\nmean |S| over seeds  (std):")
    hdr = f"{'config':>24} " + " ".join(f"g={int(g):>3}" for g in GAMMAS)
    print(hdr)
    for cname, _ in CONFIGS:
        cells = []
        for g in GAMMAS:
            s = summary[cname][str(g)]
            cells.append(f"{s['mean_size']:4.1f}")
        print(f"{cname:>24} " + "   ".join(cells))


if __name__ == "__main__":
    main()
