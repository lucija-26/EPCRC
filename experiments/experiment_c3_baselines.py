"""Experiment C3 -- joint coverage selection against standard pruning rules.

Claim C3 (plan section 5): at equal physical-panel size or equal cost, convex
coverage should preserve individual virtual judges better than top-accuracy
selection, random selection, one-per-family selection, correlation clustering,
and pairwise-distance coresets.

The comparison is run three ways, because "better" has three meanings here:

*at equal k*      every selector picks k judges and we compare held-out error;
*at equal cost*   selectors are compared at matched measured GPU seconds,
                  since judges are not equally expensive to run;
*against a floor* a rank-k subspace that may use latent directions no judge
                  occupies gives a bound no physical panel can beat.

Everything is fitted on FIT and reported on the locked TEST split, with cluster
bootstrap confidence intervals over items and a band across the five split
seeds.  Point estimates without intervals are not reportable for this claim:
the panel is small and the selectors are often within noise of each other.

Usage:

    python -u experiments/experiment_c3_baselines.py
    python -u experiments/experiment_c3_baselines.py --split-seeds all
    python -u experiments/experiment_c3_baselines.py --bootstrap 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from epcrc.judge import JudgeResponses, total_variation
from epcrc.panel import PANELS, SCORES, SPLIT_SEEDS, Panel, load_panel, scores_dir
from epcrc.reconstruction import RECONSTRUCTORS, fit_weights, score_weights
from epcrc.rewardbench import PRIMARY_SEED
from epcrc.selection_baselines import GEOMETRY_SELECTORS, lowrank_floor

from experiment_e1_compression_frontier import (  # noqa: E402
    select_backward,
    select_exhaustive,
    select_forward,
    select_one_per_family,
    select_random,
    select_top_accuracy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "c3_baselines.json")

N_RANDOM_DRAWS = 100          # section 21.1 asks for 100 random draws per k
# Plan section 34.1: 2,000 replicates during development, 10,000 for the
# final main tables once the scored outputs are cached.
N_BOOTSTRAP = 2000


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

def judge_seconds(panel: Panel, scores_dir: str) -> Dict[str, float]:
    """Measured inference seconds per judge, summed over its context blocks.

    Cached blocks report the wall time of the run that produced them, so this
    is observed cost rather than a parameter-count proxy.  Blocks that were
    served from cache report the original timing of the run that wrote them.
    """
    out: Dict[str, float] = {}
    for j in panel.judge_ids:
        total = 0.0
        for c in panel.context_names:
            with open(os.path.join(scores_dir, f"{j}__{c}.json")) as handle:
                total += float(json.load(handle).get("seconds", 0.0))
        out[j] = total
    return out


def select_cost_ascending(panel: Panel, cost: Dict[str, float]) -> Dict[int, List[int]]:
    """Section 21.1 'cheapest models first'."""
    order = sorted(range(panel.N), key=lambda j: cost[panel.judge_ids[j]])
    return {k: sorted(order[:k]) for k in range(1, panel.N + 1)}


def select_cost_descending(panel: Panel, cost: Dict[str, float]) -> Dict[int, List[int]]:
    """Section 21.1 'largest models first', using measured cost as the size proxy."""
    order = sorted(range(panel.N), key=lambda j: -cost[panel.judge_ids[j]])
    return {k: sorted(order[:k]) for k in range(1, panel.N + 1)}


# --------------------------------------------------------------------------
# held-out error with per-item resolution
# --------------------------------------------------------------------------

def fit_panel_weights(
    panel: Panel,
    kept: Sequence[int],
    rule: str = "simplex",
) -> List[np.ndarray]:
    """One weight vector per judge, all fitted on FIT.

    Fitting is by far the most expensive step -- one LP per judge -- so it is
    done once per subset and the result is reused for every evaluation split.
    """
    kept = sorted(kept)
    return [fit_weights(rule, panel.splits["FIT"], j, kept)
            for j in range(panel.N)]


def apply_weights(
    panel: Panel,
    kept: Sequence[int],
    weights: Sequence[np.ndarray],
    eval_split: str,
) -> Dict[str, np.ndarray]:
    """Per-item TV and verdict agreement for every judge, on a held-out split.

    Keeping the item axis is what makes the bootstrap possible later; the
    scalar summaries in `summarise` are just reductions of these arrays.
    """
    kept = sorted(kept)
    ev = panel.splits[eval_split]

    tv = np.empty((ev.n_contexts, ev.blocks[0].shape[0], panel.N))
    hit = np.empty_like(tv)

    for j in range(panel.N):
        w = weights[j]
        for c, block in enumerate(ev.blocks):
            recon = np.tensordot(block[:, kept, :], w, axes=([1], [0]))
            truth = block[:, j, :]
            tv[c, :, j] = total_variation(truth, recon)
            hit[c, :, j] = truth.argmax(axis=1) == recon.argmax(axis=1)

    return {"tv": tv, "hit": hit}


def summarise(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    per_judge_context = arrays["tv"].mean(axis=1)          # (contexts, judges)
    worst_context = per_judge_context.max(axis=0)          # (judges,)
    return {
        "worst_judge_worst_context_tv": float(worst_context.max()),
        "mean_judge_worst_context_tv": float(worst_context.mean()),
        "median_item_tv": float(np.median(arrays["tv"])),
        "p95_item_tv": float(np.percentile(arrays["tv"], 95)),
        "verdict_agreement": float(arrays["hit"].mean()),
    }


def _group_totals(
    arrays: Dict[str, np.ndarray], groups: np.ndarray
) -> Dict[str, np.ndarray]:
    """Collapse rows to per-base-item sums, so a replicate is a sum over groups.

    A cluster bootstrap mean is (sum of the sampled groups' sums) / (sum of
    their row counts).  Precomputing the group totals turns resampling
    variable-sized clusters into ordinary fancy indexing, which keeps the whole
    bootstrap vectorised.
    """
    n_groups = int(groups.max()) + 1
    tv, hit = arrays["tv"], arrays["hit"]

    tv_sum = np.zeros((tv.shape[0], n_groups, tv.shape[2]))
    hit_sum = np.zeros_like(tv_sum)
    for c in range(tv.shape[0]):
        np.add.at(tv_sum[c], groups, tv[c])
        np.add.at(hit_sum[c], groups, hit[c])

    counts = np.bincount(groups, minlength=n_groups).astype(float)
    return {"tv_sum": tv_sum, "hit_sum": hit_sum, "counts": counts}


def bootstrap(
    arrays: Dict[str, np.ndarray],
    n_boot: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
    chunk: int = 100,
) -> Dict[str, List[float]]:
    """Cluster bootstrap over base items (plan section 34.1).

    Two dependencies have to be respected or the intervals come out too narrow:

    * the contexts hold the *same* items under different prompts, so an item is
      resampled in every context at once; and
    * one base item can produce several comparison pairs, so all of its pairs
      move together as a single unit.

    `groups` gives the base item of each row.  Without it each row is its own
    unit, which is only correct when every item yields exactly one pair.
    """
    tv = arrays["tv"]
    if groups is None:
        groups = np.arange(tv.shape[1])

    totals = _group_totals(arrays, groups)
    tv_sum, hit_sum, counts = totals["tv_sum"], totals["hit_sum"], totals["counts"]
    n_groups = counts.size
    n_judges = tv.shape[2]
    rng = np.random.default_rng(seed)

    worst, mean_j, agree = [], [], []
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n_groups, size=(size, n_groups))
        n_rows = counts[idx].sum(axis=1)                    # (size,)

        # (contexts, size, judges)
        means = np.stack([tv_sum[c][idx].sum(axis=1) / n_rows[:, None]
                          for c in range(tv_sum.shape[0])])
        worst_context = means.max(axis=0)                   # (size, judges)
        worst.extend(worst_context.max(axis=1).tolist())
        mean_j.extend(worst_context.mean(axis=1).tolist())

        hits = np.stack([hit_sum[c][idx].sum(axis=(1, 2))
                         for c in range(hit_sum.shape[0])])
        agree.extend((hits.mean(axis=0) / (n_rows * n_judges)).tolist())
        done += size

    def ci(values: List[float]) -> Dict[str, float]:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "se": float(np.std(values, ddof=1))}

    return {
        "worst_judge_worst_context_tv": ci(worst),
        "mean_judge_worst_context_tv": ci(mean_j),
        "verdict_agreement": ci(agree),
        "n_bootstrap_units": int(n_groups),
    }


def _worst_judge_replicates(
    arrays: Dict[str, np.ndarray],
    groups: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Worst-judge worst-context TV for each replicate, on given resamples."""
    totals = _group_totals(arrays, groups)
    n_rows = totals["counts"][idx].sum(axis=1)
    means = np.stack([totals["tv_sum"][c][idx].sum(axis=1) / n_rows[:, None]
                      for c in range(totals["tv_sum"].shape[0])])
    return means.max(axis=0).max(axis=1)


def paired_bootstrap(
    arrays_a: Dict[str, np.ndarray],
    arrays_b: Dict[str, np.ndarray],
    n_boot: int,
    seed: int,
    groups: np.ndarray,
) -> Dict[str, float]:
    """Difference between two methods on the *same* resampled items (section 34.2).

    Comparing two independent intervals is the wrong test: both methods are
    scored on one set of items, so most of the sampling noise is shared and
    cancels.  Resampling the same items for both and taking the difference is
    far more sensitive than asking whether two intervals overlap.

    Returns the difference a - b, so a negative value means `a` has the lower
    error and is the better method.
    """
    rng = np.random.default_rng(seed)
    n_groups = int(groups.max()) + 1
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))

    diff = (_worst_judge_replicates(arrays_a, groups, idx)
            - _worst_judge_replicates(arrays_b, groups, idx))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "delta": float(np.mean(diff)),
        "lo": float(lo),
        "hi": float(hi),
        "prob_a_better": float((diff < 0).mean()),
        "excludes_zero": bool(hi < 0 or lo > 0),
    }


# --------------------------------------------------------------------------
# selector registry
# --------------------------------------------------------------------------

def build_selectors(
    panel: Panel,
    cost: Dict[str, float],
    scores_dir: str,
    max_exhaustive: int,
    n_random: int,
) -> Dict[str, Dict[int, List[int]]]:
    """Every competitor in the C3 comparison, keyed by name."""
    chains: Dict[str, Dict[int, List[int]]] = {
        # the method under test
        "coverage_backward": select_backward(panel),
        "coverage_forward": select_forward(panel),
        # 21.1 panel-size baselines
        "top_accuracy": select_top_accuracy(panel, scores_dir),
        "one_per_family": select_one_per_family(panel),
        "cost_ascending": select_cost_ascending(panel, cost),
        "cost_descending": select_cost_descending(panel, cost),
    }

    # 21.2 pairwise-geometry baselines
    for name, selector in GEOMETRY_SELECTORS.items():
        chains[name] = selector(panel.splits["FIT"])

    exhaustive = select_exhaustive(panel, max_exhaustive)
    if exhaustive:
        chains["exhaustive"] = exhaustive

    for d in range(n_random):
        chains[f"random_{d}"] = select_random(panel, seed=1000 + d)

    # C3 is an equal-k comparison, so a selector that quietly returns the wrong
    # number of judges would compare panels of different sizes and look better
    # or worse than it is.  Refuse to report rather than allow that.
    for name, chain in chains.items():
        for k, kept in chain.items():
            if len(set(kept)) != k:
                raise RuntimeError(
                    f"selector {name!r} returned {len(set(kept))} judges at k={k}"
                )
            if not set(kept) <= set(range(panel.N)):
                raise RuntimeError(f"selector {name!r} returned out-of-range indices")

    return chains


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

# A worker process holds one panel for the whole split seed, so the response
# tensor is loaded once rather than shipped with every task.
_WORKER: Dict[str, object] = {}


def _init_worker(seed: int, scores: str, split_seed: int) -> None:
    _WORKER["panel"] = load_panel(seed, scores, split_seed=split_seed)


def _evaluate_subset(task) -> tuple:
    """Fit one subset on FIT and score it on both held-out splits."""
    kept, n_boot, boot_seed = task
    panel: Panel = _WORKER["panel"]

    weights = fit_panel_weights(panel, kept)
    out: Dict[str, object] = {}
    for split in ("CERT", "TEST"):
        arrays = apply_weights(panel, kept, weights, split)
        out[split] = summarise(arrays)
        if split == "TEST" and n_boot:
            out["TEST_ci"] = bootstrap(
                arrays, n_boot, seed=boot_seed,
                groups=panel.item_groups.get("TEST"),
            )
    return kept, out


def run_split_seed(
    split_seed: int,
    seed: int,
    scores: str,
    max_exhaustive: int,
    n_random: int,
    n_boot: int,
    workers: int,
) -> Dict[str, object]:
    panel = load_panel(seed, scores, split_seed=split_seed)
    cost = judge_seconds(panel, scores)
    chains = build_selectors(panel, cost, scores, max_exhaustive, n_random)

    # Selectors overlap heavily -- nested chains share prefixes and random
    # draws collide at small k -- so evaluating unique subsets instead of
    # (method, k) pairs removes most of the work before any of it is done.
    unique = sorted({tuple(sorted(kept))
                     for chain in chains.values() for kept in chain.values()})
    tasks = [(kept, n_boot, split_seed + 7919 * i)
             for i, kept in enumerate(unique)]
    total_pairs = sum(len(c) for c in chains.values())
    print(f"  {total_pairs} (method, k) pairs -> {len(unique)} unique subsets, "
          f"{workers} workers", flush=True)

    evaluated: Dict[tuple, Dict[str, object]] = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(seed, scores, split_seed),
        ) as pool:
            for kept, row in pool.map(_evaluate_subset, tasks, chunksize=1):
                evaluated[kept] = row
    else:
        _init_worker(seed, scores, split_seed)
        for task in tasks:
            kept, row = _evaluate_subset(task)
            evaluated[kept] = row

    total_cost = sum(cost.values())
    methods: Dict[str, object] = {}
    for name, chain in chains.items():
        rows: Dict[str, Dict[str, object]] = {}
        for k, kept in sorted(chain.items()):
            spent = sum(cost[panel.judge_ids[j]] for j in kept)
            rows[str(k)] = {
                "kept": [panel.judge_ids[j] for j in kept],
                "k": k,
                "cost_seconds": spent,
                "cost_frac": spent / total_cost,
                "calls_avoided_frac": 1.0 - k / panel.N,
                **evaluated[tuple(sorted(kept))],
            }
        methods[name] = rows
        if not name.startswith("random_"):
            print(f"  [{name}] "
                  + " ".join(
                      f"k{k}={rows[str(k)]['TEST']['worst_judge_worst_context_tv']:.3f}"
                      for k in sorted(chain)
                  ), flush=True)

    floors = {
        kind: lowrank_floor(panel.splits["FIT"], panel.splits["TEST"], kind)
        for kind in ("pca", "nmf")
    }

    paired = run_paired_comparisons(panel, chains, n_boot, split_seed)

    return {
        "split_seed": split_seed,
        "paired_TEST": paired,
        "judges": panel.judge_ids,
        "models": panel.model_ids,
        "contexts": panel.context_names,
        "split_items": panel.n_items,
        "cost_seconds": cost,
        "n_unique_subsets": len(unique),
        "methods": methods,
        "lowrank_floor_TEST": {
            kind: {str(k): v for k, v in table.items()}
            for kind, table in floors.items()
        },
    }


def run_paired_comparisons(
    panel: Panel,
    chains: Dict[str, Dict[int, List[int]]],
    n_boot: int,
    split_seed: int,
    reference: str = "coverage_backward",
) -> Dict[str, object]:
    """Paired bootstrap of the reference method against each baseline.

    Only the non-trivial budgets are compared: at k = 1 and k = N every method
    is forced to the same answer, so a difference there would be identically
    zero and would only dilute the comparison.

    Random draws are pooled into one `random` competitor by taking the draw
    that does best at each budget, which is the honest version of "would a
    lucky random panel have done as well".
    """
    if reference not in chains:
        return {}

    groups = panel.item_groups["TEST"]
    sizes = [k for k in sorted(chains[reference]) if 1 < k < panel.N]

    def arrays_for(kept: Sequence[int]) -> Dict[str, np.ndarray]:
        weights = fit_panel_weights(panel, kept)
        return apply_weights(panel, kept, weights, "TEST")

    out: Dict[str, object] = {}
    for k in sizes:
        ref_arrays = arrays_for(chains[reference][k])
        per_method: Dict[str, object] = {}

        competitors: Dict[str, List[int]] = {}
        best_random, best_score = None, np.inf
        for name, chain in chains.items():
            if name == reference or k not in chain:
                continue
            if name.startswith("random_"):
                score = summarise(arrays_for(chain[k]))["worst_judge_worst_context_tv"]
                if score < best_score:
                    best_random, best_score = chain[k], score
                continue
            competitors[name] = chain[k]
        if best_random is not None:
            competitors["random_best_draw"] = best_random

        for name, kept in competitors.items():
            per_method[name] = paired_bootstrap(
                ref_arrays, arrays_for(kept), n_boot,
                seed=split_seed + 104729 * k, groups=groups,
            )
        out[str(k)] = per_method

    return out


def run_reconstruction_ablation(
    panel: Panel,
    chain: Dict[int, List[int]],
) -> Dict[str, object]:
    """Section 21.3: hold the selected set fixed and vary only the fitting rule."""
    out: Dict[str, object] = {}
    for k, kept in sorted(chain.items()):
        if k in (0, panel.N):
            continue
        per_rule: Dict[str, object] = {}
        for rule in RECONSTRUCTORS:
            rows = []
            for j in range(panel.N):
                w = fit_weights(rule, panel.splits["FIT"], j, kept)
                rows.append(score_weights(panel.splits["TEST"], j, kept, w))
            per_rule[rule] = {
                "worst_judge_worst_context_tv": float(
                    max(r["worst_context_mean_tv"] for r in rows)
                ),
                "mean_judge_worst_context_tv": float(
                    np.mean([r["worst_context_mean_tv"] for r in rows])
                ),
                "verdict_agreement": float(np.mean([r["verdict_agreement"] for r in rows])),
                "off_simplex_frac": float(np.mean([r["off_simplex_frac"] for r in rows])),
                "max_weight_l1": float(max(r["weight_l1"] for r in rows)),
                "min_weight": float(min(r["weight_min"] for r in rows)),
            }
        out[str(k)] = {"kept": [panel.judge_ids[j] for j in kept], "rules": per_rule}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None,
                        help="registered panel to load; overrides --scores")
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--split-seeds", default="all",
                        help="'primary', 'all', or a comma-separated list; the "
                             "section 34.4 robustness table needs all five")
    parser.add_argument("--max-exhaustive", type=int, default=10,
                        help="skip enumeration above this panel size; 0 disables it")
    parser.add_argument("--random-draws", type=int, default=N_RANDOM_DRAWS)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="processes for the subset evaluations; 1 disables them")
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(ROOT, "results", f"c3_baselines_{args.panel}.json")

    if args.split_seeds == "primary":
        seeds = [SPLIT_SEEDS[0]]
    elif args.split_seeds == "all":
        seeds = list(SPLIT_SEEDS)
    else:
        seeds = [int(s) for s in args.split_seeds.split(",")]

    started = time.time()
    per_seed = []
    for split_seed in seeds:
        print(f"[split_seed {split_seed}]", flush=True)
        per_seed.append(run_split_seed(
            split_seed, args.seed, args.scores,
            args.max_exhaustive, args.random_draws, args.bootstrap,
            args.workers,
        ))

    panel = load_panel(args.seed, args.scores, split_seed=seeds[0])
    ablation = run_reconstruction_ablation(panel, select_backward(panel))

    payload = {
        "experiment": "C3",
        "claim": "C3",
        "pairs_seed": args.seed,
        "split_seeds": seeds,
        "n_random_draws": args.random_draws,
        "n_bootstrap": args.bootstrap,
        "wall_seconds": time.time() - started,
        "per_split_seed": per_seed,
        "reconstruction_ablation_TEST": ablation,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
