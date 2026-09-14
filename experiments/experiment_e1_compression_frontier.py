"""Experiment E1 -- physical-to-virtual compression frontier (claim C2).

How small can the physical panel become while every *virtual* judge is still
reconstructed within a small worst-context error?

For each budget k the experiment picks a physical subset S with |S| = k, fits
one simplex weight vector per virtual judge on FIT, and reports the error those
weights actually achieve on CERT and on the locked TEST split.  Weights are
never fitted on the split they are scored on, which is what makes the frontier
a held-out claim rather than a restatement of the training fit.

The headline number is the worst-judge worst-context mean TV error, because C2
is a statement about *every* judge being preserved, not the average one.

Usage:

    python -u experiments/experiment_e1_compression_frontier.py
    python -u experiments/experiment_e1_compression_frontier.py --max-exhaustive 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import (
    JudgeCoverageFunctional,
    solve_minimax_weights,
    total_variation,
)
from epcrc.panel import PANELS, SCORES, Panel, load_panel, scores_dir
from epcrc.rewardbench import PRIMARY_SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e1_frontier.json")

# Section 23 fixes these tolerance points; the grid may be extended but not
# reduced.
E1_GAMMAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

N_RANDOM_DRAWS = 20


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# metrics for one subset
# --------------------------------------------------------------------------

def _verdicts(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.argmax(axis=-1)


def evaluate_subset(
    panel: Panel,
    kept: Sequence[int],
    eval_split: str,
) -> Dict[str, object]:
    """Fit weights on FIT, then score them on `eval_split`.

    Judges inside the physical subset are measured, not assumed: they are
    reconstructed by themselves at zero error, and including them in the
    aggregate is what makes the frontier comparable across budgets.
    """
    kept = sorted(kept)
    fit = panel.splits["FIT"]
    ev = panel.splits[eval_split]

    per_judge: Dict[str, Dict[str, float]] = {}
    item_errors: List[np.ndarray] = []
    agreements: List[float] = []
    margin_abs: List[np.ndarray] = []

    for j, judge_id in enumerate(panel.judge_ids):
        if j in kept:
            w = np.zeros(len(kept))
            w[kept.index(j)] = 1.0
        else:
            _, w = solve_minimax_weights(fit, j, kept)

        worst = -np.inf
        judge_items: List[np.ndarray] = []
        for block in ev.blocks:
            recon = np.tensordot(block[:, kept, :], w, axes=([1], [0]))
            truth = block[:, j, :]
            tv = total_variation(truth, recon)
            worst = max(worst, float(tv.mean()))
            judge_items.append(tv)

            agreements.append(float((_verdicts(truth) == _verdicts(recon)).mean()))
            margin_abs.append(np.abs(
                (truth[:, 0] - truth[:, 1]) - (recon[:, 0] - recon[:, 1])
            ))

        stacked = np.concatenate(judge_items)
        item_errors.append(stacked)
        per_judge[judge_id] = {
            "worst_context_mean_tv": worst,
            "median_item_tv": float(np.median(stacked)),
            "p95_item_tv": float(np.percentile(stacked, 95)),
            "in_physical_panel": j in kept,
        }

    every_item = np.concatenate(item_errors)
    return {
        "kept": [panel.judge_ids[j] for j in kept],
        "k": len(kept),
        "worst_judge_worst_context_tv": float(
            max(v["worst_context_mean_tv"] for v in per_judge.values())
        ),
        "mean_judge_worst_context_tv": float(
            np.mean([v["worst_context_mean_tv"] for v in per_judge.values()])
        ),
        "median_item_tv": float(np.median(every_item)),
        "p95_item_tv": float(np.percentile(every_item, 95)),
        "verdict_agreement": float(np.mean(agreements)),
        "signed_margin_mae": float(np.concatenate(margin_abs).mean()),
        "calls_avoided_frac": 1.0 - len(kept) / panel.N,
        "per_judge": per_judge,
    }


# --------------------------------------------------------------------------
# selection methods
# --------------------------------------------------------------------------

def select_backward(panel: Panel) -> Dict[int, List[int]]:
    """Greedy backward elimination on FIT, recording the subset at every size.

    At each step the judge whose removal leaves the smallest worst-judge error
    is dropped, so the returned chain is nested.
    """
    cov = JudgeCoverageFunctional(panel.splits["FIT"], panel.splits["FIT"],
                                  panel.judge_ids)
    current = set(range(panel.N))
    chain = {panel.N: sorted(current)}

    while len(current) > 1:
        best, best_score = None, np.inf
        for j in sorted(current):
            score, _ = cov.compute_coverage(current - {j})
            if score < best_score:
                best, best_score = j, score
        current = current - {best}
        chain[len(current)] = sorted(current)
    return chain


def select_forward(panel: Panel) -> Dict[int, List[int]]:
    """Greedy forward selection on FIT, recording the subset at every size."""
    cov = JudgeCoverageFunctional(panel.splits["FIT"], panel.splits["FIT"],
                                  panel.judge_ids)
    current: set = set()
    chain: Dict[int, List[int]] = {}

    while len(current) < panel.N:
        best, best_score = None, np.inf
        for j in range(panel.N):
            if j in current:
                continue
            score, _ = cov.compute_coverage(current | {j})
            if score < best_score:
                best, best_score = j, score
        current = current | {best}
        chain[len(current)] = sorted(current)
    return chain


def select_exhaustive(panel: Panel, max_n: int = 10) -> Dict[int, List[int]]:
    """True optimum at every budget, by enumeration. Only for small panels."""
    if panel.N > max_n:
        return {}
    cov = JudgeCoverageFunctional(panel.splits["FIT"], panel.splits["FIT"],
                                  panel.judge_ids)
    chain: Dict[int, List[int]] = {}
    for k in range(1, panel.N + 1):
        best, best_score = None, np.inf
        for combo in combinations(range(panel.N), k):
            score, _ = cov.compute_coverage(set(combo))
            if score < best_score:
                best, best_score = list(combo), score
        chain[k] = best
    return chain


def select_top_accuracy(panel: Panel, scores_dir: str) -> Dict[int, List[int]]:
    """Baseline: keep the judges that agree with the human label most often.

    `scores_dir` is required rather than defaulted: it must be the directory the
    panel itself was loaded from.  A default here would let a Core-20 panel read
    Core-8 accuracies whenever the judge ids happened to overlap, and rank the
    baseline on numbers belonging to a different run.
    """
    means = []
    for j in panel.judge_ids:
        accs = [
            json.load(open(os.path.join(scores_dir, f"{j}__{c}.json")))
            ["accuracy"]["three_class_accuracy"]
            for c in panel.context_names
        ]
        means.append(float(np.mean(accs)))
    order = list(np.argsort(means)[::-1])
    return {k: sorted(order[:k]) for k in range(1, panel.N + 1)}


def select_one_per_family(panel: Panel) -> Dict[int, List[int]]:
    """Baseline: spread the budget over distinct model families before repeating."""
    families: Dict[str, List[int]] = {}
    for j, model in enumerate(panel.model_ids):
        families.setdefault(model.split("/")[0].lower(), []).append(j)

    order: List[int] = []
    buckets = [list(v) for _, v in sorted(families.items())]
    while any(buckets):
        for bucket in buckets:
            if bucket:
                order.append(bucket.pop(0))
    return {k: sorted(order[:k]) for k in range(1, panel.N + 1)}


def select_random(panel: Panel, seed: int) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(panel.N))
    return {k: sorted(int(x) for x in order[:k]) for k in range(1, panel.N + 1)}


# --------------------------------------------------------------------------
# tolerance form
# --------------------------------------------------------------------------

def tolerance_frontier(
    panel: Panel,
    chain: Dict[int, List[int]],
    eval_rows: Dict[int, Dict[str, object]],
) -> Dict[str, object]:
    """Smallest budget on this chain whose held-out error meets each gamma."""
    out = {}
    for gamma in E1_GAMMAS:
        feasible = [
            k for k in sorted(chain)
            if eval_rows[k]["worst_judge_worst_context_tv"] <= gamma
        ]
        if feasible:
            k = min(feasible)
            out[str(gamma)] = {
                "min_k": k,
                "kept": [panel.judge_ids[j] for j in chain[k]],
            }
        else:
            out[str(gamma)] = {"min_k": None, "kept": None}
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(
    panel: Panel, max_exhaustive: int = 10, scores_dir: str = SCORES
) -> Dict[str, object]:
    methods: Dict[str, Dict[int, List[int]]] = {
        "coverage_backward": select_backward(panel),
        "coverage_forward": select_forward(panel),
        "top_accuracy": select_top_accuracy(panel, scores_dir),
        "one_per_family": select_one_per_family(panel),
    }
    exhaustive = select_exhaustive(panel, max_exhaustive)
    if exhaustive:
        methods["exhaustive"] = exhaustive
    for d in range(N_RANDOM_DRAWS):
        methods[f"random_{d}"] = select_random(panel, seed=1000 + d)

    results: Dict[str, object] = {}
    for name, chain in methods.items():
        print(f"[{name}]", flush=True)
        rows = {}
        for split in ("CERT", "TEST"):
            rows[split] = {k: evaluate_subset(panel, kept, split)
                           for k, kept in sorted(chain.items())}
        results[name] = {
            "chain": {str(k): [panel.judge_ids[j] for j in v]
                      for k, v in sorted(chain.items())},
            "CERT": {str(k): v for k, v in rows["CERT"].items()},
            "TEST": {str(k): v for k, v in rows["TEST"].items()},
            "tolerance_TEST": tolerance_frontier(panel, chain, rows["TEST"]),
        }
        for k in sorted(chain):
            print(f"  k={k}  TEST worst {rows['TEST'][k]['worst_judge_worst_context_tv']:.4f}"
                  f"  agree {rows['TEST'][k]['verdict_agreement']:.3f}", flush=True)

    return {
        "experiment": "E1",
        "claim": "C2",
        "seed": PRIMARY_SEED,
        "judges": panel.judge_ids,
        "models": panel.model_ids,
        "contexts": panel.context_names,
        "split_items": panel.n_items,
        "gammas": E1_GAMMAS,
        "n_random_draws": N_RANDOM_DRAWS,
        "methods": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None,
                        help="registered panel to load; overrides --scores")
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--max-exhaustive", type=int, default=10,
                        help="skip enumeration above this panel size; 0 disables it")
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(ROOT, "results", f"e1_frontier_{args.panel}.json")

    panel = load_panel(args.seed, args.scores)
    print(f"panel: {panel.judge_ids}")
    print(f"items per split: {panel.n_items}")

    payload = run(panel, args.max_exhaustive, args.scores)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
