"""E2 — downstream preservation (plan section 24, claim C5).

C5 asks whether aggregations computed from the *reconstructed* panel match
aggregations computed from the actual full panel: preference accuracy within a
percentage point, little movement in NLL or Brier, and rank agreement at or
above 0.95.

The comparison the claim turns on is `virtual` against `physical`, both at the
same budget k:

`full`
    all twenty real judges.  The reference every other arm is scored against;
    it is what the compression is trying to preserve, not a competitor.
`virtual`
    the k selected judges, plus a reconstruction of each of the other 20 - k
    from them.  The aggregate therefore still runs over twenty judges, but only
    k models are ever executed.
`physical`
    the same k judges, aggregated on their own.  This is what dropping judges
    looks like without reconstruction, and the gap between it and `virtual` is
    what reconstruction buys.
`top_accuracy`, `one_per_family`, `random`, `single_best`
    equal-sized physical panels chosen by the usual heuristics.

Every arm is scored on the locked TEST split under all seven contexts.
Weights and learned aggregators are fitted on FIT only.

Reported per context and averaged over the five predeclared split seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.downstream import (
    apply_logistic_aggregate,
    disagreement_rate,
    evaluate_aggregate,
    fit_logistic_aggregate,
    majority_aggregate,
    mean_aggregate,
)
from epcrc.judge import solve_minimax_weights
from epcrc.panel import (
    PANELS,
    SCORES,
    SPLIT_SEEDS,
    Panel,
    load_raw_panel,
    scores_dir,
    split_panel,
)
from epcrc.rewardbench import PRIMARY_SEED, grouped_split

from experiment_e1_compression_frontier import (
    select_backward,
    select_one_per_family,
    select_random,
    select_top_accuracy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "c5_downstream.json")

K_GRID = [4, 6, 8, 10, 12, 14, 16]


def virtual_block(panel: Panel, kept: List[int], split: str) -> List[np.ndarray]:
    """Per-context blocks for `split` with every absent judge reconstructed.

    Kept judges pass through untouched: their own row is what a deployed panel
    would actually observe, so replacing it with a fit of itself would make the
    arm easier than the thing it stands for.

    One weight vector per absent judge, fitted once on FIT against the worst
    context, then applied in every context -- the same object C2 certifies, now
    used rather than measured.
    """
    fit = panel.splits["FIT"]
    blocks = panel.splits[split].blocks
    n_contexts = len(blocks)

    weights = {
        target: solve_minimax_weights(fit, target, kept)[1]
        for target in range(panel.N)
        if target not in kept
    }

    out = [block.copy() for block in blocks]
    for target, w in weights.items():
        for c in range(n_contexts):
            out[c][:, target, :] = np.tensordot(
                blocks[c][:, kept, :], w, axes=([1], [0])
            )
    return out


def physical_block(panel: Panel, kept: List[int], split: str) -> List[np.ndarray]:
    return [block[:, kept, :] for block in panel.splits[split].blocks]


def score_arm(
    panel: Panel,
    test_blocks: List[np.ndarray],
    fit_blocks: List[np.ndarray],
    reference: Dict[int, np.ndarray],
    domain_codes: np.ndarray,
) -> Dict[str, object]:
    """Every aggregator's metrics for one arm, per context."""
    gold_fit = panel.gold["FIT"]
    gold_test = panel.gold["TEST"]

    per_context: Dict[str, Dict[str, float]] = {}
    for c, name in enumerate(panel.context_names):
        aggregates = {
            "mean": mean_aggregate(test_blocks[c]),
            "majority": majority_aggregate(test_blocks[c]),
        }
        model = fit_logistic_aggregate(fit_blocks[c], gold_fit)
        aggregates["logistic"] = apply_logistic_aggregate(model, test_blocks[c])

        for aggregator, values in aggregates.items():
            metrics = evaluate_aggregate(
                values, gold_test, reference.get((c, aggregator)), domain_codes
            )
            metrics["disagreement_rate"] = disagreement_rate(test_blocks[c])
            per_context[f"{name}|{aggregator}"] = metrics

    return per_context


def reference_aggregates(
    panel: Panel, blocks: List[np.ndarray], fit_blocks: List[np.ndarray]
) -> Dict[tuple, np.ndarray]:
    """The full panel's aggregate for every (context, aggregator)."""
    gold_fit = panel.gold["FIT"]
    out = {}
    for c in range(len(blocks)):
        out[(c, "mean")] = mean_aggregate(blocks[c])
        out[(c, "majority")] = majority_aggregate(blocks[c])
        model = fit_logistic_aggregate(fit_blocks[c], gold_fit)
        out[(c, "logistic")] = apply_logistic_aggregate(model, blocks[c])
    return out


def run_one_split(panel: Panel, scores: str, k_grid: List[int]) -> Dict[str, object]:
    domains = panel.domains["TEST"]
    domain_codes = np.unique(domains, return_inverse=True)[1]

    full_test = list(panel.splits["TEST"].blocks)
    full_fit = list(panel.splits["FIT"].blocks)
    reference = reference_aggregates(panel, full_test, full_fit)

    results: Dict[str, object] = {
        "full": {
            "k": panel.N,
            "kept": panel.judge_ids,
            "contexts": score_arm(panel, full_test, full_fit, reference, domain_codes),
        }
    }

    chains = {
        "virtual": select_backward(panel),
        "physical": select_backward(panel),
        "top_accuracy": select_top_accuracy(panel, scores),
        "one_per_family": select_one_per_family(panel),
        "random": select_random(panel, seed=4242),
    }

    for k in k_grid:
        for arm, chain in chains.items():
            if k not in chain:
                continue
            kept = list(chain[k])
            if arm == "virtual":
                test_blocks = virtual_block(panel, kept, "TEST")
                fit_blocks = virtual_block(panel, kept, "FIT")
            else:
                test_blocks = physical_block(panel, kept, "TEST")
                fit_blocks = physical_block(panel, kept, "FIT")
            results[f"{arm}@{k}"] = {
                "arm": arm,
                "k": k,
                "kept": [panel.judge_ids[j] for j in kept],
                "contexts": score_arm(
                    panel, test_blocks, fit_blocks, reference, domain_codes
                ),
            }
        print(f"  k={k} done", flush=True)

    # The single best judge is a budget of one and does not belong on the grid.
    best = int(np.argmax([
        (panel.splits["TEST"].blocks[0][:, j, :].argmax(axis=1) == panel.gold["TEST"]).mean()
        for j in range(panel.N)
    ]))
    results["single_best"] = {
        "arm": "single_best",
        "k": 1,
        "kept": [panel.judge_ids[best]],
        "contexts": score_arm(
            panel,
            physical_block(panel, [best], "TEST"),
            physical_block(panel, [best], "FIT"),
            reference,
            domain_codes,
        ),
    }
    return results


def run(raw, split_seeds: List[int], scores: str, k_grid: List[int]) -> Dict[str, object]:
    per_seed = {}
    for split_seed in split_seeds:
        print(f"[split seed {split_seed}]", flush=True)
        panel = split_panel(raw, grouped_split(raw.pairs, seed=split_seed))
        per_seed[str(split_seed)] = run_one_split(panel, scores, k_grid)

    return {
        "experiment": "E2",
        "claim": "C5",
        "seed": PRIMARY_SEED,
        "judges": raw.judge_ids,
        "models": raw.model_ids,
        "contexts": raw.context_names,
        "split_seeds": split_seeds,
        "k_grid": k_grid,
        "aggregators": ["mean", "majority", "logistic"],
        "note_system_ranking": (
            "RewardBench 2 does not record which system produced a response, so "
            "the plan's system-level ranking has no axis in this data. The "
            "reported rank metrics are over items (primary surrogate) and over "
            "domains (coarse); a true system ranking needs the JuStRank source "
            "in E7 and is not claimed here."
        ),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None)
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--split-seeds", type=int, nargs="*", default=SPLIT_SEEDS)
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(ROOT, "results", f"c5_downstream_{args.panel}.json")

    raw = load_raw_panel(args.seed, args.scores)
    print(f"panel: {raw.judge_ids}")

    payload = run(raw, args.split_seeds, args.scores, K_GRID)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
