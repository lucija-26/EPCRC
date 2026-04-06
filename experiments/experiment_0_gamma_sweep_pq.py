"""Experiment 0 (gamma sweep): priority queue pruning across multiple gamma
tolerances on the UTD19 ecosystem.

Reuses the cached per-city model bundle. For each gamma in GAMMAS, runs the
lazy greedy priority queue pruner (phase 1: forward add, phase 2: backward
cleanup), records full per-step metrics, and writes:

    results/experiment_0_sweep/priority_queue_pruning_gamma<g>.json  - per-run
    results/experiment_0_sweep/summary_all.json  - aggregate across all 3 algos

Run from project root:
    python experiments/experiment_0_gamma_sweep_pq.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from epcrc.coverage import CoverageFunctional
from epcrc.metrics import step_metrics
from epcrc.pruning import PriorityQueuePruner
from epcrc.utd19_pipeline import DEFAULT_CONFIG, build_or_load_bundle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CSV = os.path.join(_project_root, "data", "utd19_u.csv")
CACHE_DIR = os.path.join(_project_root, "data", "exp_0_utd19_cache")
OUT_DIR = os.path.join(_project_root, "results", "experiment_0_sweep")

GAMMAS = [40.0, 60.0, 80.0, 100.0]
METRIC = "mean_abs"
CITIES = None  # None = every city in the CSV


# ---------------------------------------------------------------------------
# Helpers (same pattern as experiment_0_gamma_sweep.py)
# ---------------------------------------------------------------------------
def run_algorithm(algo_name, pruner_cls, coverage_fn, gamma, model_names, N):
    """Run one pruning algorithm and return a rich records dict."""
    t0 = time.time()
    step_records = []

    # Step 0: initial snapshot (S = empty for forward-style algorithms)
    init_metrics = step_metrics(coverage_fn, set(), gamma)
    init_metrics.update(
        {
            "iteration": 0,
            "action": "init",
            "changed_model_idx": None,
            "changed_model_name": None,
            "elapsed_seconds": round(time.time() - t0, 4),
        }
    )
    step_records.append(init_metrics)

    pruner = pruner_cls(coverage_fn=coverage_fn, tolerance_gamma=gamma)
    result = pruner.run(debug=False)
    total_time = time.time() - t0

    for step in result.history:
        m = step_metrics(coverage_fn, step.kept_set, gamma)
        m.update(
            {
                "iteration": int(step.iteration),
                "action": step.action,
                "changed_model_idx": (
                    int(step.removed_model_idx)
                    if step.removed_model_idx is not None
                    else None
                ),
                "changed_model_name": step.removed_model_name,
                "elapsed_seconds": round(time.time() - t0, 4),
            }
        )
        step_records.append(m)

    final = step_records[-1]
    return {
        "algorithm": algo_name,
        "gamma": gamma,
        "wall_time_seconds": round(total_time, 4),
        "final": {
            "kept_set_size": final["kept_set_size"],
            "coverage_E_S": final["coverage_E_S"],
            "sum_uniqueness_U": final["sum_uniqueness_U"],
            "satisfies_gamma": final["satisfies_gamma"],
            "kept_set_names": final["kept_set_names"],
            "removed_set_names": final["removed_set_names"],
        },
        "steps": step_records,
    }


def main() -> None:
    W = 76
    print("=" * W)
    print("EXPERIMENT 0 (GAMMA SWEEP) - UTD19 priority queue pruning")
    print("=" * W)
    print(f"  gammas  = {GAMMAS}")
    print(f"  metric  = {METRIC}")
    print(f"  data    = {DATA_CSV}")
    print(f"  cache   = {CACHE_DIR}")
    print(f"  out_dir = {OUT_DIR}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load (or build) the cached bundle
    # ------------------------------------------------------------------
    bundle = build_or_load_bundle(
        data_csv=DATA_CSV,
        cache_dir=CACHE_DIR,
        cities=CITIES,
        verbose=True,
    )
    model_names = bundle.model_names
    N = len(model_names)
    print(f"\n[sweep] N = {N}, Y_fit = {bundle.Y_fit.shape}, Y_eval = {bundle.Y_eval.shape}\n")

    # ------------------------------------------------------------------
    # Sweep: priority queue pruning
    # ------------------------------------------------------------------
    algo_name = "priority_queue_pruning"
    pq_summary_rows = []

    for gamma in GAMMAS:
        print("=" * W)
        print(f"  gamma = {gamma}")
        print("=" * W)

        # Fresh coverage fn per run
        coverage_fn = CoverageFunctional(
            Y_fit=bundle.Y_fit,
            Y_eval=bundle.Y_eval,
            model_names=model_names,
            metric=METRIC,
        )
        run = run_algorithm(
            algo_name, PriorityQueuePruner, coverage_fn, gamma, model_names, N
        )

        # Per-run JSON
        fname = f"{algo_name}_gamma{int(gamma)}.json"
        fpath = os.path.join(OUT_DIR, fname)
        payload = {
            "algorithm": algo_name,
            "config": {
                "gamma": gamma,
                "metric": METRIC,
                "n_models": N,
                "model_names": model_names,
                "data_csv": DATA_CSV,
                "pipeline": {
                    k: v for k, v in DEFAULT_CONFIG.items() if k != "model_params"
                },
                "model_params": DEFAULT_CONFIG["model_params"],
                "per_city_train_rmse": bundle.per_city_rmse,
                "Y_fit_shape": list(bundle.Y_fit.shape),
                "Y_eval_shape": list(bundle.Y_eval.shape),
            },
            "wall_time_seconds": run["wall_time_seconds"],
            "final": run["final"],
            "steps": run["steps"],
        }
        with open(fpath, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        fin = run["final"]
        print(
            f"  [{algo_name:>25}]  |S|={fin['kept_set_size']:>2}  "
            f"E(S)={fin['coverage_E_S']:>8.3f}  "
            f"sumU={fin['sum_uniqueness_U']:>9.3f}  "
            f"gamma_ok={fin['satisfies_gamma']}  "
            f"[{run['wall_time_seconds']}s]  -> {fname}"
        )
        pq_summary_rows.append(
            {
                "gamma": gamma,
                "algorithm": algo_name,
                "kept_set_size": fin["kept_set_size"],
                "coverage_E_S": fin["coverage_E_S"],
                "sum_uniqueness_U": fin["sum_uniqueness_U"],
                "satisfies_gamma": fin["satisfies_gamma"],
                "kept_set_names": fin["kept_set_names"],
                "removed_set_names": fin["removed_set_names"],
                "wall_time_seconds": run["wall_time_seconds"],
                "result_file": fname,
            }
        )
        print()

    # ------------------------------------------------------------------
    # Build combined summary (merge with existing summary if present)
    # ------------------------------------------------------------------
    existing_summary_path = os.path.join(OUT_DIR, "summary.json")
    all_rows = []
    if os.path.exists(existing_summary_path):
        with open(existing_summary_path) as f:
            existing = json.load(f)
        # Keep rows from other algorithms, replace any existing PQ rows
        all_rows = [r for r in existing.get("runs", [])
                    if r["algorithm"] != algo_name]
    all_rows.extend(pq_summary_rows)

    # Sort by gamma then algorithm for readability
    all_rows.sort(key=lambda r: (r["gamma"], r["algorithm"]))
    all_algorithms = sorted(set(r["algorithm"] for r in all_rows))

    summary_all = {
        "experiment": "experiment_0_gamma_sweep",
        "description": (
            "Backward elimination + forward selection + priority queue pruning "
            "sweep across multiple gamma tolerances on the UTD19 traffic ecosystem."
        ),
        "gammas": GAMMAS,
        "algorithms": all_algorithms,
        "metric": METRIC,
        "n_models": N,
        "model_names": model_names,
        "runs": all_rows,
    }
    summary_all_path = os.path.join(OUT_DIR, "summary_all.json")
    with open(summary_all_path, "w") as f:
        json.dump(summary_all, f, indent=2, default=str)
    print(f"[io] summary_all -> {summary_all_path}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("COMPARISON TABLE (all algorithms)")
    print("=" * W)
    print(
        f"  {'gamma':>6s}  {'algorithm':>25s}  {'|S|':>4s}  "
        f"{'E(S)':>10s}  {'sum U':>10s}  {'time (s)':>10s}"
    )
    print(
        f"  {'-'*6}  {'-'*25}  {'-'*4}  "
        f"{'-'*10}  {'-'*10}  {'-'*10}"
    )
    for r in all_rows:
        print(
            f"  {r['gamma']:>6.0f}  {r['algorithm']:>25s}  {r['kept_set_size']:>4d}  "
            f"{r['coverage_E_S']:>10.3f}  {r['sum_uniqueness_U']:>10.3f}  "
            f"{r['wall_time_seconds']:>10.2f}"
        )

    # ------------------------------------------------------------------
    # Kept-set comparison per gamma
    # ------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("KEPT-SET COMPARISON PER GAMMA")
    print("=" * W)
    for gamma in GAMMAS:
        gamma_rows = [r for r in all_rows if r["gamma"] == gamma]
        print(f"\n  gamma = {gamma:.0f}:")
        for r in gamma_rows:
            algo_short = r["algorithm"].replace("_", " ")
            print(f"    {algo_short:>25s}: {sorted(r['kept_set_names'])}")

    print(f"\n{'=' * W}")
    print("SWEEP COMPLETE")
    print("=" * W)


if __name__ == "__main__":
    main()
