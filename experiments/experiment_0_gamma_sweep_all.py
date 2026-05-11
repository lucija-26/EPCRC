"""Experiment 0 (gamma sweep): backward elimination, forward selection, and
PriorityQueue+kswap pruning on the UTD19 ecosystem.

For each gamma in GAMMAS, runs all three algorithms, records full per-step
metrics, and writes:

    results/experiment_0_sweep_all/<algo>_gamma<g>.json  - per-run trajectory
    results/experiment_0_sweep_all/summary_all.json      - combined table
    results/experiment_0_sweep_all/README.md             - human-readable summary

Run from project root:
    python experiments/experiment_0_gamma_sweep_all.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import partial

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from epcrc.coverage import CoverageFunctional
from epcrc.metrics import step_metrics
from epcrc.pruning import (
    BackwardEliminationPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)
from epcrc.utd19_pipeline import DEFAULT_CONFIG, build_or_load_bundle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CSV = os.path.join(_project_root, "data", "utd19_u.csv")
CACHE_DIR = os.path.join(_project_root, "data", "exp_0_utd19_cache")
OUT_DIR = os.path.join(_project_root, "results", "experiment_0_sweep_all")

GAMMAS = [40.0, 60.0, 80.0, 100.0]
METRIC = "mean_abs"
CITIES = None  # None = every city in the CSV

# k-swap settings for PriorityQueuePruner
KSWAP_MAX_K = 2
KSWAP_MODE = "first"       # "first" | "best"
KSWAP_MAX_CANDIDATES = None  # None = no cap; set e.g. 500 to limit runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_y_stats(Y_eval: np.ndarray, model_names: list) -> dict:
    N = Y_eval.shape[1]
    per_col = []
    for i, name in enumerate(model_names):
        c = Y_eval[:, i]
        per_col.append(
            {
                "model": name,
                "mean": float(c.mean()),
                "std": float(c.std()),
                "min": float(c.min()),
                "max": float(c.max()),
            }
        )
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.abs(Y_eval[:, i] - Y_eval[:, j]).mean()
    off = D[~np.eye(N, dtype=bool)]
    return {
        "global": {
            "mean": float(Y_eval.mean()),
            "std": float(Y_eval.std()),
            "abs_mean": float(np.abs(Y_eval).mean()),
        },
        "pairwise_abs_diff": {
            "mean": float(off.mean()),
            "median": float(np.median(off)),
            "min": float(off.min()),
            "max": float(off.max()),
        },
        "per_column": per_col,
    }


def run_algorithm(
    algo_name: str,
    pruner_factory,
    coverage_fn: CoverageFunctional,
    gamma: float,
    init_snapshot_set: set,
) -> dict:
    """Run one pruning algorithm and return a step-trajectory dict.

    `pruner_factory` is a callable (class or partial) that accepts
    (coverage_fn, tolerance_gamma) and returns a pruner with a .run() method.
    `init_snapshot_set` is the set used for the step-0 metrics snapshot —
    full set for backward, empty set for forward/PQ.
    """
    t0 = time.time()
    N = coverage_fn.N

    step_records = []
    snap0 = step_metrics(coverage_fn, init_snapshot_set, gamma)
    snap0.update(
        {
            "iteration": 0,
            "action": "init",
            "changed_model_idx": None,
            "changed_model_name": None,
            "elapsed_seconds": 0.0,
        }
    )
    step_records.append(snap0)

    pruner = pruner_factory(coverage_fn=coverage_fn, tolerance_gamma=gamma)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    W = 82
    print("=" * W)
    print("EXPERIMENT 0 (GAMMA SWEEP ALL) - backward / forward / PQ+kswap")
    print("=" * W)
    print(f"  gammas           = {GAMMAS}")
    print(f"  metric           = {METRIC}")
    print(f"  kswap_max_k      = {KSWAP_MAX_K}")
    print(f"  kswap_mode       = {KSWAP_MODE}")
    print(f"  kswap_max_cands  = {KSWAP_MAX_CANDIDATES}")
    print(f"  data             = {DATA_CSV}")
    print(f"  out_dir          = {OUT_DIR}")
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

    y_stats = compute_y_stats(bundle.Y_eval, model_names)
    print(
        f"[Y scale] global mean={y_stats['global']['mean']:.2f}  "
        f"std={y_stats['global']['std']:.2f}"
    )
    print(
        f"[Y scale] pairwise |Yi-Yj| mean={y_stats['pairwise_abs_diff']['mean']:.2f}  "
        f"median={y_stats['pairwise_abs_diff']['median']:.2f}"
    )
    print()

    # Algorithm registry: (label, factory, init_snapshot_set)
    pq_kswap_factory = partial(
        PriorityQueuePruner,
        max_swap_k=KSWAP_MAX_K,
        improvement_mode=KSWAP_MODE,
        max_candidates=KSWAP_MAX_CANDIDATES,
    )
    algorithms = [
        ("backward_elimination", BackwardEliminationPruner, set(range(N))),
        ("forward_selection",    ForwardSelectionPruner,    set()),
        ("pq_kswap",             pq_kswap_factory,          set()),
    ]

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    summary_rows = []
    config_block = {
        "metric": METRIC,
        "n_models": N,
        "model_names": model_names,
        "data_csv": DATA_CSV,
        "pipeline": {k: v for k, v in DEFAULT_CONFIG.items() if k != "model_params"},
        "model_params": DEFAULT_CONFIG["model_params"],
        "per_city_train_rmse": bundle.per_city_rmse,
        "Y_fit_shape": list(bundle.Y_fit.shape),
        "Y_eval_shape": list(bundle.Y_eval.shape),
        "kswap_max_k": KSWAP_MAX_K,
        "kswap_improvement_mode": KSWAP_MODE,
        "kswap_max_candidates": KSWAP_MAX_CANDIDATES,
    }

    for gamma in GAMMAS:
        print("=" * W)
        print(f"  gamma = {gamma}")
        print("=" * W)

        for algo_name, factory, init_snap in algorithms:
            coverage_fn = CoverageFunctional(
                Y_fit=bundle.Y_fit,
                Y_eval=bundle.Y_eval,
                model_names=model_names,
                metric=METRIC,
            )
            run = run_algorithm(algo_name, factory, coverage_fn, gamma, init_snap)

            fname = f"{algo_name}_gamma{int(gamma)}.json"
            fpath = os.path.join(OUT_DIR, fname)
            with open(fpath, "w") as f:
                json.dump(
                    {
                        "algorithm": algo_name,
                        "config": dict(config_block, gamma=gamma),
                        "wall_time_seconds": run["wall_time_seconds"],
                        "final": run["final"],
                        "steps": run["steps"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

            fin = run["final"]
            print(
                f"  [{algo_name:>22}]  |S|={fin['kept_set_size']:>2}  "
                f"E(S)={fin['coverage_E_S']:>8.3f}  "
                f"sumU={fin['sum_uniqueness_U']:>9.3f}  "
                f"ok={fin['satisfies_gamma']}  "
                f"[{run['wall_time_seconds']:.1f}s]  -> {fname}"
            )
            summary_rows.append(
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
    # Summary JSON
    # ------------------------------------------------------------------
    summary_all = {
        "experiment": "experiment_0_gamma_sweep_all",
        "description": (
            "Backward elimination + forward selection + PQ-kswap pruning "
            "sweep across multiple gamma tolerances on the UTD19 ecosystem."
        ),
        "gammas": GAMMAS,
        "algorithms": [a[0] for a in algorithms],
        **config_block,
        "Y_eval_stats": y_stats,
        "runs": sorted(summary_rows, key=lambda r: (r["gamma"], r["algorithm"])),
    }
    summary_path = os.path.join(OUT_DIR, "summary_all.json")
    with open(summary_path, "w") as f:
        json.dump(summary_all, f, indent=2, default=str)
    print(f"[io] summary -> {summary_path}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    algo_order = [a[0] for a in algorithms]
    print(f"\n{'=' * W}")
    print("COMPARISON TABLE")
    print("=" * W)
    print(
        f"  {'gamma':>6}  {'algorithm':>22}  {'|S|':>4}  "
        f"{'E(S)':>10}  {'sum U':>10}  {'time(s)':>8}"
    )
    print(f"  {'-'*6}  {'-'*22}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}")
    for gamma in GAMMAS:
        for algo in algo_order:
            r = next(
                x for x in summary_rows
                if x["gamma"] == gamma and x["algorithm"] == algo
            )
            print(
                f"  {r['gamma']:>6.0f}  {r['algorithm']:>22}  {r['kept_set_size']:>4}  "
                f"{r['coverage_E_S']:>10.3f}  {r['sum_uniqueness_U']:>10.3f}  "
                f"{r['wall_time_seconds']:>8.2f}"
            )
        print()

    # ------------------------------------------------------------------
    # Kept-set comparison per gamma
    # ------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("KEPT-SET COMPARISON PER GAMMA")
    print("=" * W)
    for gamma in GAMMAS:
        rows_g = {
            r["algorithm"]: r
            for r in summary_rows
            if r["gamma"] == gamma
        }
        rb = rows_g.get("backward_elimination")
        rf = rows_g.get("forward_selection")
        rp = rows_g.get("pq_kswap")
        bset = set(rb["kept_set_names"]) if rb else set()
        fset = set(rf["kept_set_names"]) if rf else set()
        pset = set(rp["kept_set_names"]) if rp else set()
        print(f"\n  gamma = {gamma:.0f}:")
        print(f"    {'backward':>22} ({len(bset)}): {sorted(bset)}")
        print(f"    {'forward':>22} ({len(fset)}): {sorted(fset)}")
        print(f"    {'pq_kswap':>22} ({len(pset)}): {sorted(pset)}")
        print(f"    pq_kswap saves vs backward: {sorted(bset - pset)}")
        print(f"    pq_kswap saves vs forward:  {sorted(fset - pset)}")

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    lines = [
        "# Experiment 0 - All-Algorithm Gamma Sweep",
        "",
        "Backward elimination, forward selection, and PQ+kswap pruning "
        f"run across gamma ∈ {GAMMAS} on the UTD19 ecosystem (N={N} models).",
        "",
        f"k-swap config: max_k={KSWAP_MAX_K}, mode={KSWAP_MODE!r}, "
        f"max_candidates={KSWAP_MAX_CANDIDATES}.",
        "",
        "## Y scale (gamma interpretation)",
        "",
        f"- global mean = **{y_stats['global']['mean']:.2f}**, "
        f"std = **{y_stats['global']['std']:.2f}**",
        f"- pairwise mean |Yi−Yj| = **{y_stats['pairwise_abs_diff']['mean']:.2f}** "
        f"(median {y_stats['pairwise_abs_diff']['median']:.2f})",
        "",
        "## Results",
        "",
        "| gamma | algorithm | \\|S\\| | E(S) | sum U | time(s) |",
        "|---|---|---|---|---|---|",
    ]
    for gamma in GAMMAS:
        for algo in algo_order:
            r = next(
                x for x in summary_rows
                if x["gamma"] == gamma and x["algorithm"] == algo
            )
            lines.append(
                f"| {r['gamma']:.0f} | {r['algorithm']} | {r['kept_set_size']} | "
                f"{r['coverage_E_S']:.3f} | {r['sum_uniqueness_U']:.3f} | "
                f"{r['wall_time_seconds']:.1f} |"
            )
    lines += [
        "",
        "## Kept-set comparison",
        "",
    ]
    for gamma in GAMMAS:
        rows_g = {r["algorithm"]: r for r in summary_rows if r["gamma"] == gamma}
        bset = set(rows_g.get("backward_elimination", {}).get("kept_set_names", []))
        fset = set(rows_g.get("forward_selection", {}).get("kept_set_names", []))
        pset = set(rows_g.get("pq_kswap", {}).get("kept_set_names", []))
        lines += [
            f"### gamma = {gamma:.0f}",
            "",
            f"- backward ({len(bset)}): {sorted(bset)}",
            f"- forward  ({len(fset)}): {sorted(fset)}",
            f"- pq_kswap ({len(pset)}): {sorted(pset)}",
            f"- pq_kswap saves vs backward: {sorted(bset - pset)}",
            f"- pq_kswap saves vs forward:  {sorted(fset - pset)}",
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `summary_all.json` — aggregate table + Y stats + pipeline config",
    ]
    for r in sorted(summary_rows, key=lambda x: (x["gamma"], x["algorithm"])):
        lines.append(f"- `{r['result_file']}` — full trajectory")
    lines.append("")

    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[io] readme  -> {readme_path}")

    print(f"\n{'=' * W}")
    print("SWEEP COMPLETE")
    print("=" * W)


if __name__ == "__main__":
    main()
