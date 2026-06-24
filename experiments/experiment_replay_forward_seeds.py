"""Replay the synthetic forward-search seeds and run forward/backward/kswap.

This reproduces the RNG sequence used in `experiment_forward_beats_backward.py`
and runs three pruners on the same instances and gamma grid:
  - ForwardSelectionPruner
  - BackwardEliminationPruner
  - PriorityQueuePruner (kswap)

Outputs a single JSON suitable for plotting:
  results/forward_beats_backward/replay_results.json

Run (example):
  python experiments/experiment_replay_forward_seeds.py  --n-seeds 400 --kswap-k 2

You should run this on the server (it can be large: ~100k runs for defaults).
Use --n-seeds small for quick tests.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import (
    BackwardEliminationPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)


OUT_DIR = os.path.join(_root, "results", "forward_beats_backward")
OUT_PATH = os.path.join(OUT_DIR, "replay_results.json")


def mean_pairwise_dist(pts: np.ndarray) -> float:
    n = pts.shape[0]
    tot = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += float(np.linalg.norm(pts[i] - pts[j]))
            cnt += 1
    return tot / max(cnt, 1)


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Replay forward seeds and run pruners on same instances")
    p.add_argument("--n-seeds", type=int, default=400, help="number of seeds per (N,dim) to replay")
    p.add_argument("--n-gamma", type=int, default=25, help="gamma grid resolution per instance")
    p.add_argument("--kswap-k", type=int, default=2, help="k for k-swap PriorityQueuePruner")
    p.add_argument("--n-range", nargs="*", type=int, default=[8, 9, 10, 11, 12], help="Ns to try")
    p.add_argument("--dims", nargs="*", type=int, default=[2, 3], help="dimensions to try")
    p.add_argument("--metric", default="mean_abs")
    p.add_argument("--max-instances", type=int, default=0, help="stop after this many instances (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rng_master = np.random.default_rng(0)
    records: List[dict] = []
    instance_count = 0

    for dim in args.dims:
        for N in args.n_range:
            for seed in range(args.n_seeds):
                if args.max_instances and instance_count >= args.max_instances:
                    break
                # reproduce the same per-instance RNG used by the forward search
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                pts = rng.standard_normal((N, dim))
                Y = pts.T.copy()
                model_names = [f"m{i}" for i in range(N)]

                cov = CoverageFunctional(Y_fit=Y, Y_eval=Y, model_names=model_names, metric=args.metric)

                scale = mean_pairwise_dist(pts)
                gammas = np.linspace(0.02, 0.6, args.n_gamma) * scale

                for g in gammas:
                    fwd = ForwardSelectionPruner(cov, float(g)).run()
                    bwd = BackwardEliminationPruner(cov, float(g)).run()
                    pq = PriorityQueuePruner(cov, float(g), max_swap_k=args.kswap_k).run()

                    rec = {
                        "dim": int(dim),
                        "N": int(N),
                        "seed": int(seed),
                        "gamma": float(g),
                        "gamma_over_scale": float(g / scale),
                        "n_forward": int(len(fwd.kept_set)),
                        "n_backward": int(len(bwd.kept_set)),
                        "n_kswap": int(len(pq.kept_set)),
                        "forward_kept": sorted(int(x) for x in fwd.kept_set),
                        "backward_kept": sorted(int(x) for x in bwd.kept_set),
                        "kswap_kept": sorted(int(x) for x in pq.kept_set),
                        "E_forward": float(fwd.coverage),
                        "E_backward": float(bwd.coverage),
                        "E_kswap": float(pq.coverage),
                    }
                    records.append(rec)

                instance_count += 1

    payload = {
        "config": {
            "n_seeds": args.n_seeds,
            "n_range": args.n_range,
            "dims": args.dims,
            "n_gamma": args.n_gamma,
            "metric": args.metric,
            "kswap_k": args.kswap_k,
        },
        "n_records": len(records),
        "records": records,
    }

    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote {OUT_PATH} with {len(records)} records")


if __name__ == "__main__":
    main()
