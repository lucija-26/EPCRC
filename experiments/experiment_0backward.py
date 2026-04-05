"""Experiment 0 (backward): backward elimination on the real UTD19 ecosystem.

Reuses the cached per-city models and response matrices produced by
`experiment_0forward.py` (or trains them on first call) and runs Section 4.1
backward elimination. At every iteration we log a rich metrics snapshot
(|S|, E(S), sum U, per-model U, routing sparsity, ...) to JSON.

Run from project root:
    python experiments/experiment_0backward.py
"""

from __future__ import annotations

import json
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from epcrc.coverage import CoverageFunctional
from epcrc.metrics import step_metrics
from epcrc.pruning import BackwardEliminationPruner
from epcrc.utd19_pipeline import DEFAULT_CONFIG, build_or_load_bundle


# ---------------------------------------------------------------------------
# Configuration (must match experiment_0forward so the bundle is reused)
# ---------------------------------------------------------------------------
DATA_CSV = os.path.join(_project_root, "data", "utd19_u.csv")
CACHE_DIR = os.path.join(_project_root, "data", "exp_0_utd19_cache")
OUT_PATH = os.path.join(_project_root, "results", "experiment_0backward.json")

GAMMA = 40.0
METRIC = "mean_abs"
CITIES = None  # None = use every city available in the CSV


def main() -> None:
    W = 76
    print("=" * W)
    print("EXPERIMENT 0 (BACKWARD ELIMINATION) - UTD19 real traffic ecosystem")
    print("=" * W)
    print(f"  gamma   = {GAMMA}")
    print(f"  metric  = {METRIC}")
    print(f"  data    = {DATA_CSV}")
    print(f"  cache   = {CACHE_DIR}")
    print(f"  output  = {OUT_PATH}")
    print()

    # ------------------------------------------------------------------
    # Load trained-model bundle (created or reused via shared cache)
    # ------------------------------------------------------------------
    bundle = build_or_load_bundle(
        data_csv=DATA_CSV,
        cache_dir=CACHE_DIR,
        cities=CITIES,
        verbose=True,
    )
    model_names = bundle.model_names
    N = len(model_names)
    print(f"\n[pruning] ecosystem size N = {N}")
    print(f"[pruning] Y_fit = {bundle.Y_fit.shape}  Y_eval = {bundle.Y_eval.shape}\n")

    # ------------------------------------------------------------------
    # Coverage + pruner
    # ------------------------------------------------------------------
    coverage_fn = CoverageFunctional(
        Y_fit=bundle.Y_fit,
        Y_eval=bundle.Y_eval,
        model_names=model_names,
        metric=METRIC,
    )

    # Always include step 0 (S = full ecosystem) for a full trajectory
    t0 = time.time()
    step_records = []
    init_metrics = step_metrics(coverage_fn, set(range(N)), GAMMA)
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

    # Run backward elimination
    be_pruner = BackwardEliminationPruner(coverage_fn=coverage_fn, tolerance_gamma=GAMMA)
    result = be_pruner.run(debug=False)
    total_time = time.time() - t0

    print(f"[pruning] finished in {total_time:.2f}s, {len(result.history)} history entries\n")
    print(f"  {'iter':>4s} {'action':6s} {'model':24s} {'|S|':>4s} {'E(S)':>10s} {'sum U':>12s}")
    print(f"  {'----':>4s} {'------':6s} {'-----':24s} {'---':>4s} {'----':>10s} {'-----':>12s}")
    print(
        f"  {0:>4d} {'INIT':6s} {'(full ecosystem)':24s} "
        f"{init_metrics['kept_set_size']:>4d} "
        f"{init_metrics['coverage_E_S']:>10.4f} "
        f"{init_metrics['sum_uniqueness_U']:>12.4f}"
    )

    for step in result.history:
        m = step_metrics(coverage_fn, step.kept_set, GAMMA)
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
        name = step.removed_model_name or "(stop)"
        print(
            f"  {step.iteration:>4d} {step.action.upper():6s} {name[:24]:24s} "
            f"{m['kept_set_size']:>4d} "
            f"{m['coverage_E_S']:>10.4f} "
            f"{m['sum_uniqueness_U']:>12.4f}"
        )

    final = step_records[-1]
    print()
    print("=" * W)
    print("  BACKWARD ELIMINATION SUMMARY")
    print("=" * W)
    print(f"  |S|               = {final['kept_set_size']}")
    print(f"  E(S)              = {final['coverage_E_S']:.4f}")
    print(f"  sum_uniqueness U  = {final['sum_uniqueness_U']:.4f}")
    print(f"  satisfies gamma   = {final['satisfies_gamma']}")
    print(f"  wall-time (s)     = {total_time:.2f}")
    print(f"  kept cities       = {final['kept_set_names']}")
    print("=" * W)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    payload = {
        "algorithm": "backward_elimination",
        "config": {
            "gamma": GAMMA,
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
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[io] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
