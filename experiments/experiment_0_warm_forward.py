"""Task 3: UTD19 warm-forward gamma sweep with random subset runs.

Compares three methods on random model subsets drawn from the cached UTD19 bundle:
    - backward (BackwardEliminationPruner)
    - forward_cold (ForwardSelectionPruner from empty set)
    - forward_warm_worst (WarmStartForwardWorstCoveredPruner)

Warm seed set per subset:
    M = {2 farthest-apart models in that subset by mean_abs distance on Y_eval}

This mirrors the subset-robustness JSON structure so plotting in notebooks is easy.

Run from project root:
    python experiments/experiment_0_warm_forward.py [n_seeds] [gammas_csv] [subset_size]
Example:
    python experiments/experiment_0_warm_forward.py 20 "40,60,80,100,120,140,160" 15
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import (
    BackwardEliminationPruner,
    ForwardSelectionPruner,
    WarmStartForwardWorstCoveredPruner,
)
from epcrc.utd19_pipeline import build_or_load_bundle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CSV = os.path.join(_project_root, "data", "utd19_u.csv")
CACHE_DIR = os.path.join(_project_root, "data", "exp_0_utd19_cache")
OUT_DIR = os.path.join(_project_root, "results", "warm_forward_utd19")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
GAMMAS = (
    [float(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
)
SUBSET_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 15
METRIC = "mean_abs"
CITIES = None


def farthest_pair_mean_abs(Y_eval: np.ndarray) -> Tuple[int, int, float]:
    """Return (i, j, d_ij) maximizing mean absolute difference on Y_eval columns."""
    N = Y_eval.shape[1]
    best_i, best_j, best_d = 0, 1, -float("inf")
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.abs(Y_eval[:, i] - Y_eval[:, j]).mean())
            if d > best_d:
                best_i, best_j, best_d = i, j, d
    return best_i, best_j, best_d


def run_one(
    algo_name: str,
    coverage_fn: CoverageFunctional,
    gamma: float,
    seed_set: set | None = None,
) -> Dict:
    t0 = time.time()

    if algo_name == "backward":
        res = BackwardEliminationPruner(coverage_fn, gamma).run(debug=False)
    elif algo_name == "forward_cold":
        res = ForwardSelectionPruner(coverage_fn, gamma).run(debug=False)
    elif algo_name == "forward_warm_worst":
        res = WarmStartForwardWorstCoveredPruner(coverage_fn, gamma, seed_set=seed_set).run(debug=False)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return {
        "algorithm": algo_name,
        "gamma": float(gamma),
        "kept_size": int(len(res.kept_set)),
        "kept_set": sorted(int(i) for i in res.kept_set),
        "kept_names": [coverage_fn.model_names[i] for i in sorted(res.kept_set)],
        "coverage": float(res.coverage),
        "sum_uniqueness": float(res.sum_uniqueness),
        "satisfies_gamma": bool(res.coverage <= gamma),
        "n_steps": int(len(res.history)),
        "wall_time_seconds": round(time.time() - t0, 4),
    }


def aggregate(records: List[Dict], gammas: List[float], configs: List[str]) -> Dict:
    """Subset-robustness style summary[cfg][gamma_str] for notebook plotting."""
    out: Dict[str, Dict[str, Dict]] = {cfg: {} for cfg in configs}
    for cfg in configs:
        for g in gammas:
            rows = [r for r in records if r["config"] == cfg and r["gamma"] == g]
            sizes = np.array([r["kept_size"] for r in rows], dtype=float)
            covs = np.array([r["coverage_E"] for r in rows], dtype=float)
            feas = np.array([r["feasible"] for r in rows], dtype=float)
            out[cfg][str(g)] = {
                "mean_size": float(sizes.mean()),
                "std_size": float(sizes.std()),
                "min_size": int(sizes.min()),
                "max_size": int(sizes.max()),
                "mean_coverage_E": float(covs.mean()),
                "std_coverage_E": float(covs.std()),
                "feasible_rate": float(feas.mean()),
                "n_seeds": int(len(rows)),
            }
    return out


def print_size_table(summary: Dict, gammas: List[float], configs: List[str]) -> None:
    print("\nmean |S| over seeds:")
    hdr = f"{'config':>20} " + " ".join(f"g={int(g):>3}" for g in gammas)
    print(hdr)
    for cfg in configs:
        vals = [summary[cfg][str(g)]["mean_size"] for g in gammas]
        print(f"{cfg:>20} " + "   ".join(f"{v:4.1f}" for v in vals))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if SUBSET_SIZE < 2:
        raise ValueError("SUBSET_SIZE must be >= 2 (warm seed needs a farthest pair)")
    if len(GAMMAS) == 0:
        raise ValueError("GAMMAS must be non-empty")

    print("=" * 82)
    print("TASK 3 - UTD19 warm-forward subset sweep")
    print("=" * 82)
    print(f"seeds   = {N_SEEDS}")
    print(f"subset  = {SUBSET_SIZE}")
    print(f"gammas  = {GAMMAS}")
    print(f"metric  = {METRIC}")
    print(f"bundle  = {os.path.join(CACHE_DIR, 'bundle.npz')}")
    print(f"out_dir = {OUT_DIR}")
    print()

    bundle = build_or_load_bundle(
        data_csv=DATA_CSV,
        cache_dir=CACHE_DIR,
        cities=CITIES,
        verbose=True,
    )

    model_names_full = list(bundle.model_names)
    N = len(model_names_full)
    if SUBSET_SIZE > N:
        raise ValueError(f"SUBSET_SIZE={SUBSET_SIZE} > N={N}")

    configs = ["backward", "forward_cold", "forward_warm_worst"]
    records: List[Dict] = []
    t_start = time.time()

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        sub = sorted(int(i) for i in rng.choice(N, size=SUBSET_SIZE, replace=False))
        sub_names = [model_names_full[i] for i in sub]

        Y_fit_sub = bundle.Y_fit[:, sub]
        Y_eval_sub = bundle.Y_eval[:, sub]
        i_far, j_far, d_far = farthest_pair_mean_abs(Y_eval_sub)
        seed_set_local = {int(i_far), int(j_far)}

        print(
            f"[seed {seed:02d}] subset_size={SUBSET_SIZE} "
            f"warm_seed=({sub_names[i_far]}, {sub_names[j_far]}) d={d_far:.2f}"
        )

        cov = CoverageFunctional(
            Y_fit=Y_fit_sub,
            Y_eval=Y_eval_sub,
            model_names=sub_names,
            metric=METRIC,
        )

        for g in GAMMAS:
            for cfg in configs:
                seed_for_cfg = seed_set_local if cfg == "forward_warm_worst" else None
                run = run_one(cfg, cov, float(g), seed_set=seed_for_cfg)
                kept_local = run["kept_set"]

                rec = {
                    "seed": int(seed),
                    "subset_global_idx": sub,
                    "subset_names": sub_names,
                    "gamma": float(g),
                    "config": cfg,
                    "kept_size": int(run["kept_size"]),
                    "coverage_E": float(run["coverage"]),
                    "feasible": bool(run["satisfies_gamma"]),
                    "kept_local_idx": kept_local,
                    "kept_global_idx": [sub[i] for i in kept_local],
                    "kept_names": run["kept_names"],
                    "sum_uniqueness_U": float(run["sum_uniqueness"]),
                    "n_steps": int(run["n_steps"]),
                    "wall_time_s": float(run["wall_time_seconds"]),
                    "warm_seed_local_idx": sorted(seed_set_local),
                    "warm_seed_global_idx": [sub[i] for i in sorted(seed_set_local)],
                    "warm_seed_names": [sub_names[i] for i in sorted(seed_set_local)],
                    "warm_seed_mean_abs_distance": float(d_far),
                }
                records.append(rec)

    summary = aggregate(records, GAMMAS, configs)
    print_size_table(summary, GAMMAS, configs)

    out = {
        "experiment": "experiment_0_warm_forward_subset_sweep",
        "config": {
            "data_csv": DATA_CSV,
            "cache_dir": CACHE_DIR,
            "metric": METRIC,
            "gammas": GAMMAS,
            "n_models_total": N,
            "subset_size": SUBSET_SIZE,
            "n_seeds": N_SEEDS,
            "configs": configs,
            "warm_seed_rule": "2 farthest-apart models in subset by mean_abs(Y_eval[:,i]-Y_eval[:,j])",
        },
        "model_names_full": model_names_full,
        "summary": summary,
        "records": records,
        "total_wall_time_s": round(time.time() - t_start, 3),
    }

    out_path = os.path.join(OUT_DIR, f"warm_forward_{N_SEEDS}seeds.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n[json] wrote {out_path}")


if __name__ == "__main__":
    main()
