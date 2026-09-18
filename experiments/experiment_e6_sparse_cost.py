"""Experiment E6 -- sparse certificates and cost-aware panels (optional claim C8).

Section 28 has two halves and they ask different questions about the same panel.

**Sparse certificates.** A virtual judge is reconstructed from a convex
combination of the retained judges, and nothing so far has limited how many of
them it leans on.  If the reconstruction needs all of S then every retained
judge has to be queried for every virtual judge, and the saving is only in the
panel, not in the serving path.  If it needs two or three, the virtual judge can
be served from a handful of calls.  The experiment caps the support at
r in {1, 2, 3, 4, infinity} and measures what the cap costs.

The l0 cap makes the inner fit combinatorial, and section 28 allows exact
enumeration for small r and |S|.  That is what is used here: for a cap of r, the
best support of size exactly r is found by trying all of them, which also covers
every smaller support because a weight vector on r judges may put zeros on some
of them.  This is an exact answer, not a relaxation, so the reported coverage is
a certificate in the same sense as everywhere else in the study.

**Cost-aware panels.** Section 28 asks for four objectives over the same
feasible family: cardinality, measured cost, memory, and deployment count.  The
feasible family is enumerated once by the backbone search, which by monotonicity
finds every subset meeting the tolerance down to the frontier, so each objective
is minimised exactly rather than greedily.  Each winning panel is then priced
under all four objectives, because the interesting number is not that the cheap
panel is cheap but what the cardinality-optimal panel costs in seconds.

Usage:

    python -u experiments/experiment_e6_sparse_cost.py --panel core20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeResponses, solve_minimax_weights, worst_context_error
from epcrc.panel import PANELS, SCORES, SPLIT_SEEDS, Panel, load_panel, scores_dir
from epcrc.rewardbench import PRIMARY_SEED
from experiments.experiment_backbone import enumerate_optima
from experiments.experiment_c3_baselines import judge_seconds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e6_sparse_cost.json")

SUPPORT_CAPS = [1, 2, 3, 4]          # section 28; infinity is the uncapped fit
WEIGHT_FLOOR = 1e-6                  # below this a weight is not a real call

# Panel sizes the sparse half is run at.  These are the budgets C2 is stated
# over, plus one larger panel so the effect of the cap can be seen when there is
# more to choose from.
SPARSE_BUDGETS = [6, 8, 10, 15]

# Tolerances the cost-aware half is run at.  Only the loose end of the section
# 23 grid admits any subset at all on this panel, so tighter points would make
# every objective pick the full panel and say nothing.
COST_GAMMAS = [0.15, 0.20, 0.25, 0.30]

# Parameter counts in billions, written down rather than parsed out of the model
# name.  "mini" and "Nemo" carry no number, and a silent mis-parse would move the
# memory ranking without anything failing.  Sources are the model cards.
PARAMS_B = {
    "Qwen/Qwen2.5-3B-Instruct": 3.09,
    "Qwen/Qwen2.5-7B-Instruct": 7.62,
    "Qwen/Qwen2.5-14B-Instruct": 14.8,
    "Qwen/Qwen3-4B": 4.02,
    "Qwen/Qwen3-8B": 8.19,
    "Qwen/Qwen3-14B": 14.8,
    "meta-llama/Llama-3.2-3B-Instruct": 3.21,
    "meta-llama/Llama-3.1-8B-Instruct": 8.03,
    "google/gemma-3-4b-it": 4.30,
    "google/gemma-3-12b-it": 12.2,
    "microsoft/Phi-4-mini-instruct": 3.84,
    "microsoft/phi-4": 14.7,
    "microsoft/Phi-3.5-mini-instruct": 3.82,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.25,
    "mistralai/Mistral-Nemo-Instruct-2407": 12.2,
    "ibm-granite/granite-3.3-8b-instruct": 8.17,
    "tiiuae/Falcon3-7B-Instruct": 7.46,
    "tiiuae/Falcon3-10B-Instruct": 10.3,
    "CohereLabs/aya-expanse-8b": 8.03,
    "allenai/OLMo-2-1124-7B-Instruct": 7.30,
}

BYTES_PER_PARAM = 2.0                # bf16 weights


# --------------------------------------------------------------------------
# sparse certificates
# --------------------------------------------------------------------------

_W: Dict[str, object] = {}


def _sparse_init(fit_blocks, ev_blocks, names) -> None:
    _W["fit"] = JudgeResponses(fit_blocks, names)
    _W["ev"] = JudgeResponses(ev_blocks, names)


def _best_support(task: Tuple[int, Tuple[int, ...], int]):
    """Best support of size r for one target, by enumeration.

    Weights are fitted on FIT and the reported error is scored on the evaluation
    split, so a support that only looks good on the fitting data cannot win.
    """
    target, kept, r = task
    fit, ev = _W["fit"], _W["ev"]
    best = (float("inf"), None, None)
    for support in combinations(kept, r):
        _, w = solve_minimax_weights(fit, target, list(support))
        error = worst_context_error(ev, target, list(support), w)
        if error < best[0]:
            best = (error, support, w)
    error, support, w = best
    keep = [int(j) for j, weight in zip(support, w) if weight > WEIGHT_FLOOR]
    return target, r, float(error), list(support), [float(x) for x in w], keep


def sparse_certificates(
    panel: Panel,
    kept: Sequence[int],
    eval_split: str,
    caps: Sequence[int],
    workers: int,
) -> Dict[str, object]:
    """Coverage as a function of the support cap, at one panel."""
    kept = tuple(sorted(kept))
    fit, ev = panel.splits["FIT"], panel.splits[eval_split]
    targets = [j for j in range(panel.N)]

    # Uncapped fit first: it is the r = infinity row and it also supplies the
    # average and maximum support size the claim is about.
    uncapped: Dict[int, Dict[str, object]] = {}
    for j in targets:
        if j in kept:
            uncapped[j] = {"error": 0.0, "support": [panel.judge_ids[j]],
                           "support_size": 1, "self": True}
            continue
        _, w = solve_minimax_weights(fit, j, list(kept))
        used = [panel.judge_ids[c] for c, weight in zip(kept, w) if weight > WEIGHT_FLOOR]
        uncapped[j] = {
            "error": float(worst_context_error(ev, j, list(kept), w)),
            "support": used,
            "support_size": len(used),
            "self": False,
        }

    tasks = [(j, kept, r) for r in caps for j in targets if j not in kept]
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_sparse_init,
            initargs=(fit.blocks, ev.blocks, fit.context_names),
        ) as pool:
            results = list(pool.map(_best_support, tasks, chunksize=1))
    else:
        _sparse_init(fit.blocks, ev.blocks, fit.context_names)
        results = [_best_support(t) for t in tasks]

    by_cap: Dict[int, Dict[int, Dict[str, object]]] = {r: {} for r in caps}
    for target, r, error, support, w, keep in results:
        by_cap[r][target] = {
            "error": error,
            "support": [panel.judge_ids[c] for c in support],
            "used": [panel.judge_ids[c] for c in keep],
            "weights": w,
        }

    rows = []
    for r in caps:
        errors = []
        sizes = []
        for j in targets:
            if j in kept:
                errors.append(0.0)
                sizes.append(1)
            else:
                errors.append(by_cap[r][j]["error"])
                sizes.append(len(by_cap[r][j]["used"]))
        rows.append({
            "r": r,
            "worst_judge_worst_context_tv": float(max(errors)),
            "mean_judge_worst_context_tv": float(np.mean(errors)),
            "mean_support_size": float(np.mean(sizes)),
            "max_support_size": int(max(sizes)),
        })

    sizes = [v["support_size"] for v in uncapped.values()]
    errors = [v["error"] for v in uncapped.values()]
    rows.append({
        "r": None,                     # the r = infinity row
        "worst_judge_worst_context_tv": float(max(errors)),
        "mean_judge_worst_context_tv": float(np.mean(errors)),
        "mean_support_size": float(np.mean(sizes)),
        "max_support_size": int(max(sizes)),
    })

    return {
        "k": len(kept),
        "kept": [panel.judge_ids[j] for j in kept],
        "by_r": rows,
        "uncapped_per_judge": {panel.judge_ids[j]: v for j, v in uncapped.items()},
        "capped_per_judge": {
            str(r): {panel.judge_ids[j]: v for j, v in by_cap[r].items()}
            for r in caps
        },
    }


def support_stability(per_seed: Dict[int, Dict[str, object]], cap: int) -> Dict[str, object]:
    """How often the same judges are chosen for a support when the split moves.

    Section 28 asks for stability of the selected supports.  A support that is
    redrawn every time the partition changes is a property of the sample, and
    reporting it as an inference path would be wrong.
    """
    seeds = list(per_seed)
    judges = sorted({
        j for seed in per_seed.values()
        for j in seed["capped_per_judge"][str(cap)]
    })
    out = {}
    for j in judges:
        picks = [
            frozenset(per_seed[s]["capped_per_judge"][str(cap)][j]["used"])
            for s in seeds if j in per_seed[s]["capped_per_judge"][str(cap)]
        ]
        if not picks:
            continue
        counts: Dict[FrozenSet[str], int] = {}
        for p in picks:
            counts[p] = counts.get(p, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])
        shared = set.intersection(*[set(p) for p in picks]) if picks else set()
        out[j] = {
            "n_seeds": len(picks),
            "modal_support": sorted(best[0]),
            "modal_count": best[1],
            "always_used": sorted(shared),
            "n_distinct_supports": len(counts),
        }
    return out


# --------------------------------------------------------------------------
# cost-aware panels
# --------------------------------------------------------------------------

def cost_models(panel: Panel, seconds: Dict[str, float]) -> Dict[str, object]:
    """The four objectives section 28 names, as functions of a judge subset."""
    missing = [m for m in panel.model_ids if m not in PARAMS_B]
    if missing:
        raise KeyError(f"no parameter count recorded for {missing}")

    params = {j: PARAMS_B[m] for j, m in zip(panel.judge_ids, panel.model_ids)}
    family = {j: m.split("/")[0] for j, m in zip(panel.judge_ids, panel.model_ids)}

    def names(subset) -> List[str]:
        return [panel.judge_ids[j] for j in sorted(subset)]

    return {
        # Every judge counts once: the objective the rest of the study minimises.
        "cardinality": lambda s: float(len(s)),
        # Measured wall time of the inference that produced the cached blocks.
        "seconds": lambda s: float(sum(seconds[j] for j in names(s))),
        # Weights of every judge in the panel held at once, which is the
        # constraint when the panel is served concurrently.
        "memory_gb": lambda s: float(
            sum(params[j] for j in names(s)) * BYTES_PER_PARAM),
        # One served endpoint per model family, so judges from a family already
        # deployed are close to free to add.
        "deployments": lambda s: float(len({family[j] for j in names(s)})),
    }


def cost_aware_panels(
    panel: Panel,
    search: Dict[str, object],
    gammas: Sequence[float],
    seconds: Dict[str, float],
) -> List[Dict[str, object]]:
    """Minimise each objective exactly over the enumerated feasible family."""
    objectives = cost_models(panel, seconds)
    levels: Dict[int, Dict[FrozenSet[int], float]] = search["levels"]

    out = []
    for gamma in gammas:
        feasible = [
            (s, e) for level in levels.values() for s, e in level.items()
            if e <= gamma
        ]
        if not feasible:
            out.append({"gamma": gamma, "feasible": False})
            continue

        row: Dict[str, object] = {"gamma": gamma, "feasible": True,
                                  "n_feasible_panels": len(feasible)}
        for name, f in objectives.items():
            # Ties are broken by the remaining objectives in a fixed order so the
            # winner does not depend on dictionary iteration order.
            best = min(feasible, key=lambda pair: (
                f(pair[0]),
                objectives["cardinality"](pair[0]),
                objectives["seconds"](pair[0]),
                sorted(pair[0]),
            ))
            subset, error = best
            row[name] = {
                "panel": [panel.judge_ids[j] for j in sorted(subset)],
                "coverage": error,
                "cost": {o: g(subset) for o, g in objectives.items()},
            }
        out.append(row)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None)
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--eval-split", default="TEST", choices=["CERT", "TEST"],
                        help="split the sparse certificates are scored on")
    parser.add_argument("--split-seeds", type=int, nargs="*", default=SPLIT_SEEDS)
    parser.add_argument("--budgets", type=int, nargs="*", default=SPARSE_BUDGETS)
    parser.add_argument("--gammas", type=float, nargs="*", default=COST_GAMMAS)
    parser.add_argument("--caps", type=int, nargs="*", default=SUPPORT_CAPS)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(ROOT, "results",
                                    f"e6_sparse_cost_{args.panel}.json")

    with open(os.path.join(ROOT, "results",
                           f"e1_frontier_{args.panel or 'core20'}.json")) as handle:
        chain = json.load(handle)["methods"]["coverage_backward"]["chain"]

    seconds = judge_seconds(
        load_panel(args.seed, args.scores), args.scores)

    sparse: Dict[str, Dict[str, object]] = {}
    per_seed_for_stability: Dict[int, Dict[int, Dict[str, object]]] = {}
    cost: Dict[str, object] = {}

    for split_seed in args.split_seeds:
        panel = load_panel(args.seed, args.scores, split_seed=split_seed)
        index = {name: i for i, name in enumerate(panel.judge_ids)}
        print(f"[split seed {split_seed}]", flush=True)

        per_budget: Dict[int, Dict[str, object]] = {}
        for k in args.budgets:
            kept = [index[n] for n in chain[str(k)]]
            start = time.time()
            per_budget[k] = sparse_certificates(
                panel, kept, args.eval_split, args.caps, args.workers)
            rows = {r["r"]: r for r in per_budget[k]["by_r"]}
            print(f"  k={k} ({time.time() - start:.0f}s) "
                  + "  ".join(
                      f"r={'inf' if r is None else r}:"
                      f"{rows[r]['worst_judge_worst_context_tv']:.3f}"
                      for r in list(args.caps) + [None]),
                  flush=True)
        sparse[str(split_seed)] = {str(k): v for k, v in per_budget.items()}
        per_seed_for_stability[split_seed] = per_budget

        if split_seed == args.split_seeds[0]:
            search = enumerate_optima(
                panel, "FIT", max(args.gammas), True, 60000, args.workers)
            cost = {
                "split_seed": split_seed,
                "by_gamma": cost_aware_panels(panel, search, args.gammas, seconds),
            }
            for row in cost["by_gamma"]:
                if row["feasible"]:
                    print(f"  gamma={row['gamma']}: "
                          f"cardinality {len(row['cardinality']['panel'])} judges / "
                          f"{row['cardinality']['cost']['seconds']:.0f}s vs "
                          f"cheapest {len(row['seconds']['panel'])} judges / "
                          f"{row['seconds']['cost']['seconds']:.0f}s", flush=True)

    stability = {
        str(k): {
            str(cap): support_stability(
                {s: v[k] for s, v in per_seed_for_stability.items()}, cap)
            for cap in args.caps
        }
        for k in args.budgets
    }

    payload = {
        "experiment": "E6",
        "plan_section": "28",
        "claim": "C8",
        "seed": args.seed,
        "eval_split": args.eval_split,
        "split_seeds": list(args.split_seeds),
        "budgets": list(args.budgets),
        "caps": list(args.caps),
        "gammas": list(args.gammas),
        "weight_floor": WEIGHT_FLOOR,
        "judge_seconds": seconds,
        "params_billion": PARAMS_B,
        "bytes_per_param": BYTES_PER_PARAM,
        "sparse": sparse,
        "support_stability": stability,
        "cost_aware": cost,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
