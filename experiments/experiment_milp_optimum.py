"""MILP oracle-routing optimum on UTD19 subsets (Open Problem 4 / whiteboard OP4).

Mirrors experiment_exact_optimum.py's subset generation (same seeds -> same
model subsets) so the MILP lower bound can be joined per-record with the
protocol-true exhaustive optimum and the greedy sizes:

    |OPT_milp(oracle)| <= |OPT_protocol(exhaustive)| <= |greedy|

The MILP picks routing weights directly against Y_eval (an oracle router),
whereas the protocol fits weights on Y_fit -- see epcrc/milp.py for the exact
semantics.  The gap between the two optima is the price of honest splitting.

Run from project root:
    python experiments/experiment_milp_optimum.py [n_seeds] [gammas_csv] [subset_size]

Defaults match experiment_exact_optimum.py.  N=15 subsets are fine for HiGHS;
for the full N=20 ecosystem (where exhaustive search is out of reach) run
    python experiments/experiment_milp_optimum.py 1 "60,80,100,120,140,160" 20
(seed irrelevant at subset_size=N).  Use the server + Gurobi if HiGHS is slow.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.milp import milp_min_representative_set

BUNDLE = os.path.join(_root, "data", "exp_0_utd19_cache", "bundle.npz")
OUT_DIR = os.path.join(_root, "results", "exact_optimum")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GAMMAS = (
    [float(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
)
SUBSET_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 15
TIME_LIMIT_S = 600.0


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(BUNDLE, allow_pickle=True)
    Y_eval = d["Y_eval"]
    model_names = [str(x) for x in d["model_names"]]
    N = len(model_names)
    if SUBSET_SIZE > N:
        raise ValueError(f"SUBSET_SIZE={SUBSET_SIZE} > N={N}")

    print(f"N={N}  subset_size={SUBSET_SIZE}  seeds={N_SEEDS}  gammas={GAMMAS}")

    records: List[dict] = []
    t_start = time.time()
    for seed in range(N_SEEDS):
        # Same RNG protocol as experiment_exact_optimum.py -> identical subsets.
        rng = np.random.default_rng(seed)
        sub = sorted(int(i) for i in rng.choice(N, size=SUBSET_SIZE, replace=False))
        sub_names = [model_names[i] for i in sub]
        Ye = Y_eval[:, sub]

        for g in sorted(GAMMAS, reverse=True):
            res = milp_min_representative_set(Ye, g, time_limit=TIME_LIMIT_S)
            records.append({
                "seed": seed,
                "subset_global_idx": sub,
                "gamma": g,
                "milp_size": res.size,
                "milp_set_names": [sub_names[i] for i in res.kept_set],
                "milp_status": res.status,
                "milp_gap": res.mip_gap,
                "solve_time_s": round(res.solve_time_s, 2),
            })
            print(
                f"  seed {seed} gamma {g:6.0f}: |S|={res.size}  "
                f"({res.solve_time_s:.1f}s, status={res.status})"
            )

    out = os.path.join(OUT_DIR, f"milp_optimum_{N_SEEDS}seeds_sub{SUBSET_SIZE}.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "milp_oracle_optimum",
            "config": {
                "n_models_total": N,
                "subset_size": SUBSET_SIZE,
                "n_seeds": N_SEEDS,
                "gammas": sorted(GAMMAS),
                "metric": "mean_abs",
                "semantics": "oracle routing on Y_eval (lower bound on protocol optimum)",
            },
            "records": records,
        }, f, indent=2)
    print(f"\n[io] wrote {out}  ({len(records)} records, {time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()
