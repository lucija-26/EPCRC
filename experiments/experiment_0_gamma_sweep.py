"""Experiment 0 (gamma sweep): backward + forward pruning across multiple gamma
tolerances on the UTD19 ecosystem.

Reuses the cached per-city model bundle produced by experiment_0forward.py /
experiment_0backward.py. For each gamma in GAMMAS, runs both backward
elimination (Section 4.1) and forward selection (Section 4.2), records full
per-step metrics, and writes:

    results/experiment_0_sweep/<algo>_gamma<g>.json   - per-run rich trajectory
    results/experiment_0_sweep/summary.json           - aggregate table + Y stats
    results/experiment_0_sweep/README.md              - plain-English summary

Run from project root:
    python experiments/experiment_0_gamma_sweep.py
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
from epcrc.pruning import BackwardEliminationPruner, ForwardSelectionPruner
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
# Helpers
# ---------------------------------------------------------------------------
def run_algorithm(algo_name, pruner_cls, coverage_fn, gamma, model_names, N):
    """Run one pruning algorithm and return a rich records dict."""
    t0 = time.time()
    step_records = []

    # Step 0: full ecosystem snapshot (for trajectory completeness)
    init_metrics = step_metrics(coverage_fn, set(range(N)), gamma)
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


def compute_y_stats(Y_eval, model_names):
    """Return scale statistics that contextualize gamma."""
    N = Y_eval.shape[1]
    per_col = []
    for i, n in enumerate(model_names):
        c = Y_eval[:, i]
        per_col.append(
            {
                "model": n,
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


def main() -> None:
    W = 76
    print("=" * W)
    print("EXPERIMENT 0 (GAMMA SWEEP) - UTD19 backward + forward")
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

    # Y-scale statistics (saved to summary for gamma interpretation)
    y_stats = compute_y_stats(bundle.Y_eval, model_names)
    print(f"[Y scale] global mean={y_stats['global']['mean']:.2f}  "
          f"std={y_stats['global']['std']:.2f}")
    print(f"[Y scale] pairwise |Y_i-Y_j| mean={y_stats['pairwise_abs_diff']['mean']:.2f}  "
          f"median={y_stats['pairwise_abs_diff']['median']:.2f}")
    print()

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    all_runs = []
    summary_rows = []

    for gamma in GAMMAS:
        print("=" * W)
        print(f"  gamma = {gamma}")
        print("=" * W)

        for algo_name, pruner_cls in [
            ("backward_elimination", BackwardEliminationPruner),
            ("forward_selection", ForwardSelectionPruner),
        ]:
            # Fresh coverage fn per run (cache correctness, no cross-talk)
            coverage_fn = CoverageFunctional(
                Y_fit=bundle.Y_fit,
                Y_eval=bundle.Y_eval,
                model_names=model_names,
                metric=METRIC,
            )
            run = run_algorithm(
                algo_name, pruner_cls, coverage_fn, gamma, model_names, N
            )
            all_runs.append(run)

            # Per-run JSON (full trajectory + config)
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
                f"  [{algo_name:>22}]  |S|={fin['kept_set_size']:>2}  "
                f"E(S)={fin['coverage_E_S']:>8.3f}  "
                f"sumU={fin['sum_uniqueness_U']:>9.3f}  "
                f"gamma_ok={fin['satisfies_gamma']}  "
                f"[{run['wall_time_seconds']}s]  -> {fname}"
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
    summary = {
        "experiment": "experiment_0_gamma_sweep",
        "description": (
            "Backward elimination + forward selection sweep across multiple "
            "gamma tolerances on the UTD19 traffic ecosystem. One JSON per "
            "(algorithm, gamma) with full step metrics; this summary holds "
            "the aggregate comparison table and Y scale reference."
        ),
        "gammas": GAMMAS,
        "algorithms": ["backward_elimination", "forward_selection"],
        "metric": METRIC,
        "n_models": N,
        "model_names": model_names,
        "data_csv": DATA_CSV,
        "cache_dir": CACHE_DIR,
        "pipeline": {k: v for k, v in DEFAULT_CONFIG.items() if k != "model_params"},
        "model_params": DEFAULT_CONFIG["model_params"],
        "per_city_train_rmse": bundle.per_city_rmse,
        "Y_fit_shape": list(bundle.Y_fit.shape),
        "Y_eval_shape": list(bundle.Y_eval.shape),
        "Y_eval_stats": y_stats,
        "runs": summary_rows,
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[io] summary -> {summary_path}")

    # ------------------------------------------------------------------
    # Human-readable README
    # ------------------------------------------------------------------
    lines = []
    lines.append("# Experiment 0 - Gamma Sweep")
    lines.append("")
    lines.append(
        "Backward elimination (Section 4.1) and forward selection (Section 4.2) "
        f"run across gamma in {GAMMAS} on the UTD19 ecosystem (N={N} city "
        "traffic-forecasting models)."
    )
    lines.append("")
    lines.append("## Y scale (for gamma interpretation)")
    lines.append("")
    lines.append(f"- Y_eval global mean = **{y_stats['global']['mean']:.2f}**, "
                 f"std = **{y_stats['global']['std']:.2f}**")
    lines.append(f"- pairwise mean |Y_i - Y_j| = **{y_stats['pairwise_abs_diff']['mean']:.2f}** "
                 f"(median {y_stats['pairwise_abs_diff']['median']:.2f}, "
                 f"min {y_stats['pairwise_abs_diff']['min']:.2f}, "
                 f"max {y_stats['pairwise_abs_diff']['max']:.2f})")
    lines.append("")
    lines.append("## Results table")
    lines.append("")
    lines.append("| gamma | algorithm | |S| | E(S) | sum U | wall (s) |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary_rows:
        algo_short = "backward" if r["algorithm"] == "backward_elimination" else "forward"
        lines.append(
            f"| {r['gamma']:.0f} | {algo_short} | {r['kept_set_size']} | "
            f"{r['coverage_E_S']:.3f} | {r['sum_uniqueness_U']:.3f} | "
            f"{r['wall_time_seconds']} |"
        )
    lines.append("")
    lines.append("## Kept-set comparison per gamma")
    lines.append("")
    for gamma in GAMMAS:
        rb = next(r for r in summary_rows
                  if r["gamma"] == gamma and r["algorithm"] == "backward_elimination")
        rf = next(r for r in summary_rows
                  if r["gamma"] == gamma and r["algorithm"] == "forward_selection")
        bset = set(rb["kept_set_names"])
        fset = set(rf["kept_set_names"])
        lines.append(f"### gamma = {gamma:.0f}")
        lines.append("")
        lines.append(f"- backward kept ({len(bset)}): {sorted(bset)}")
        lines.append(f"- forward kept  ({len(fset)}): {sorted(fset)}")
        lines.append(f"- in backward only: {sorted(bset - fset)}")
        lines.append(f"- in forward only:  {sorted(fset - bset)}")
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `summary.json` - aggregate table, Y stats, pipeline config")
    for r in summary_rows:
        lines.append(f"- `{r['result_file']}` - full trajectory (every iteration's metrics)")
    lines.append("")
    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[io] readme  -> {readme_path}")
    print()
    print("=" * W)
    print("SWEEP COMPLETE")
    print("=" * W)


if __name__ == "__main__":
    main()
