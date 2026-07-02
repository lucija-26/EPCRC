"""Exact optimum vs greedy/k-swap/MILP on the synthetic replay instances.

Regenerates the same RNG instances as experiment_replay_forward_seeds.py and,
on a sample of (instance, gamma) rows from the replay parquet, computes:

  - opt          : protocol-true minimum |S| (exhaustive, k = 1..ub)
  - backward     : plain backward elimination (from the parquet)
  - fwd_kswap2   : forward-seeded PriorityQueuePruner k-swap, k=2 (from parquet)
  - bwd_kswap2/3 : BackwardKSwapPruner with max_swap_k = 2 and 3
  - milp         : oracle-routing MILP optimum (fit == eval here, so this is
                   also the "does any certificate exist" optimum)

Sampling: N_RANDOM uniformly random rows plus ALL rows where plain backward
beat forward-seeded k-swap (the known-hard cases).  Output rows carry a
`was_worse_case` flag so unbiased statistics can be computed on the random
subsample only.

Run from project root (local-friendly; ~5-10 min for defaults):
    python experiments/experiment_synthetic_optimum.py [n_random] [kswap_ks_csv]

Output: results/exact_optimum/synthetic_optimum.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.milp import milp_min_representative_set
from epcrc.pruning import BackwardKSwapPruner

REPLAY_PARQUET = os.path.join(
    _root, "results", "forward_beats_backward", "replay_results.parquet"
)
OUT_PATH = os.path.join(_root, "results", "exact_optimum", "synthetic_optimum.json")

N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 300
KSWAP_KS = (
    [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 3]
)


def gen_instances(n_seeds=400, n_range=(8, 9, 10, 11, 12), dims=(2, 3)):
    """Identical RNG protocol to experiment_replay_forward_seeds.py."""
    rng_master = np.random.default_rng(0)
    for dim in dims:
        for N in n_range:
            for seed in range(n_seeds):
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                yield dim, N, seed, rng.standard_normal((N, dim))


def exact_opt(cov: CoverageFunctional, gamma: float, ub: int, N: int) -> int:
    for k in range(1, ub):
        for combo in combinations(range(N), k):
            E, _ = cov.compute_coverage(set(combo))
            if E <= gamma:
                return k
    return ub


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = pd.read_parquet(REPLAY_PARQUET)

    worse = df[df.n_kswap > df.n_backward]
    rng = np.random.default_rng(1)
    sample = df.iloc[rng.choice(len(df), size=N_RANDOM, replace=False)]
    targets = pd.concat([worse, sample]).drop_duplicates(
        subset=["dim", "N", "seed", "gamma"]
    )
    worse_keys = set(zip(worse.dim, worse.N, worse.seed, worse.gamma))
    need = set(zip(targets.dim, targets.N, targets.seed))
    print(f"replay rows: {len(df)}  hard cases: {len(worse)}  sampled rows: {len(targets)}")

    rows: List[dict] = []
    t0 = time.time()
    for dim, N, seed, pts in gen_instances():
        if (dim, N, seed) not in need:
            continue
        Y = pts.T.copy()
        cov = CoverageFunctional(Y, Y, [f"m{i}" for i in range(N)], metric="mean_abs")
        sub = targets[(targets.dim == dim) & (targets.N == N) & (targets.seed == seed)]
        for _, r in sub.iterrows():
            g = float(r.gamma)
            sizes: Dict[str, int] = {}
            for k in KSWAP_KS:
                res = BackwardKSwapPruner(cov, g, max_swap_k=k).run()
                sizes[f"bwd_kswap{k}"] = len(res.kept_set)
            ub = min(int(r.n_backward), int(r.n_kswap), *sizes.values())
            opt = exact_opt(cov, g, ub, N)
            m = milp_min_representative_set(Y, g)
            rows.append({
                "dim": int(dim), "N": int(N), "seed": int(seed), "gamma": g,
                "opt": int(opt),
                "milp": int(m.size),
                "backward": int(r.n_backward),
                "fwd_kswap2": int(r.n_kswap),
                **sizes,
                "was_worse_case": (dim, N, seed, r.gamma) in worse_keys,
            })

    out = {
        "experiment": "synthetic_optimum",
        "config": {
            "n_random": N_RANDOM,
            "kswap_ks": KSWAP_KS,
            "replay_source": os.path.relpath(REPLAY_PARQUET, _root),
        },
        "records": rows,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[io] wrote {OUT_PATH}  ({len(rows)} rows, {time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    for label, part in [("all sampled", res), ("random only", res[~res.was_worse_case])]:
        print(f"\n{label} (n={len(part)}):")
        for m in ["milp", "backward", "fwd_kswap2"] + [f"bwd_kswap{k}" for k in KSWAP_KS]:
            gap = part[m] - part.opt
            print(f"  {m:12s} mean gap {gap.mean():+.4f}  frac == opt {(gap == 0).mean():.4f}")


if __name__ == "__main__":
    main()
