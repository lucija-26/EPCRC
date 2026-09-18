"""Experiment E7 -- cross-benchmark transfer (plan section 29).

Does a physical basis chosen on RewardBench 2 stay useful on a benchmark it was
never selected for?  The transfer set is JudgeBench, whose pairs were built by
somebody else, so a transfer gap cannot be an artefact of this project's pair
builder.

Three settings, exactly as the plan states them:

1. **frozen basis, frozen weights** -- select S and fit every virtual judge's
   weights on RewardBench FIT, then apply both, unchanged, to JudgeBench TEST;
2. **frozen basis, refitted weights** -- keep S, refit only the weights on a
   calibration sample of JudgeBench FIT, evaluate on JudgeBench TEST;
3. **oracle reselection** -- reselect S on JudgeBench FIT.  This is a diagnostic
   upper bound and is never reported as transfer.

Setting 3 is also what the retained-set overlap is measured against: if the
basis chosen in domain is close to the basis the transfer benchmark would have
chosen for itself, the basis is the part that generalises.

Two limits of this transfer set have to travel with every number it produces.
JudgeBench contains no ties, so nothing here says whether the tie corner
survives compression; and it has 620 pairs against RewardBench's several
thousand, so the calibration curve runs out of data before it flattens.

Usage:

    python -u experiments/experiment_e7_transfer.py --panel core20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.downstream import evaluate_aggregate, mean_aggregate
from epcrc.judge import (
    JudgeCoverageFunctional,
    JudgeResponses,
    solve_minimax_weights,
    total_variation,
)
from epcrc.panel import PANELS, Panel, load_panel, scores_dir
from epcrc.rewardbench import PRIMARY_SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Budgets worth reporting.  Below 4 every method is so far from usable that the
# comparison between them carries no information, and above 15 the panel is
# barely compressed.
BUDGETS = [4, 6, 8, 10, 12, 15]

# Calibration sizes for setting 2, in pairs.  310 is the whole of JudgeBench
# FIT, so the last point is the most refitting can ever buy.
CALIBRATION_SIZES = [10, 25, 50, 100, 200, 310]

N_RANDOM_DRAWS = 5


# --------------------------------------------------------------------------
# weights and scoring
# --------------------------------------------------------------------------

def _rows(responses: JudgeResponses, rows: np.ndarray) -> JudgeResponses:
    return JudgeResponses([b[rows] for b in responses.blocks], responses.context_names)


def fit_basis_weights(
    fit: JudgeResponses, kept: Sequence[int], n_judges: int
) -> Dict[int, np.ndarray]:
    """One worst-context simplex weight vector per judge, fitted on `fit`.

    Judges inside the basis get the indicator weight rather than a solved one:
    they are reconstructed by themselves, at zero error, on any benchmark.
    """
    kept = list(kept)
    weights: Dict[int, np.ndarray] = {}
    for j in range(n_judges):
        if j in kept:
            w = np.zeros(len(kept))
            w[kept.index(j)] = 1.0
        else:
            _, w = solve_minimax_weights(fit, j, kept)
        weights[j] = w
    return weights


def reconstruct(ev: JudgeResponses, kept: Sequence[int],
                weights: Dict[int, np.ndarray], n_judges: int) -> List[np.ndarray]:
    """The whole panel as the basis sees it: one (items, judges, 3) block per context."""
    kept = list(kept)
    return [
        np.stack(
            [np.tensordot(block[:, kept, :], weights[j], axes=([1], [0]))
             for j in range(n_judges)],
            axis=1,
        )
        for block in ev.blocks
    ]


def score(
    panel: Panel,
    split: str,
    kept: Sequence[int],
    weights: Dict[int, np.ndarray],
) -> Dict[str, object]:
    """Reconstruction error and downstream preservation on one split."""
    ev = panel.splits[split]
    recon_blocks = reconstruct(ev, kept, weights, panel.N)

    per_judge_worst = []
    agreements = []
    for j in range(panel.N):
        worst = -np.inf
        for block, recon in zip(ev.blocks, recon_blocks):
            tv = total_variation(block[:, j, :], recon[:, j, :])
            worst = max(worst, float(tv.mean()))
            agreements.append(float(
                (block[:, j, :].argmax(axis=-1) == recon[:, j, :].argmax(axis=-1)).mean()
            ))
        per_judge_worst.append(worst)

    out: Dict[str, object] = {
        "worst_judge_worst_context_tv": float(max(per_judge_worst)),
        "mean_judge_worst_context_tv": float(np.mean(per_judge_worst)),
        "verdict_agreement": float(np.mean(agreements)),
    }

    gold = panel.gold.get(split)
    if gold is not None:
        domains = panel.domains[split]
        groups = np.unique(domains, return_inverse=True)[1]
        # The clean context is the deployed one; the rest are interventions.
        clean = ev.blocks[0]
        reference = mean_aggregate(clean)
        candidate = mean_aggregate(recon_blocks[0])
        out["downstream"] = evaluate_aggregate(candidate, gold, reference, groups)
        out["downstream_full_panel_accuracy"] = evaluate_aggregate(
            reference, gold
        )["accuracy"]

    return out


# --------------------------------------------------------------------------
# basis selection
# --------------------------------------------------------------------------

def backward_chain(fit: JudgeResponses, judge_ids: Sequence[str]) -> Dict[int, List[int]]:
    """Greedy backward elimination, recording the basis at every size."""
    cov = JudgeCoverageFunctional(fit, fit, list(judge_ids))
    current = set(range(len(judge_ids)))
    chain = {len(current): sorted(current)}
    while len(current) > 1:
        best, best_score = None, np.inf
        for j in sorted(current):
            value, _ = cov.compute_coverage(current - {j})
            if value < best_score:
                best, best_score = j, value
        current = current - {best}
        chain[len(current)] = sorted(current)
    return chain


def top_accuracy_chain(panel: Panel, scores: str) -> Dict[int, List[int]]:
    means = []
    for j in panel.judge_ids:
        accs = [
            json.load(open(os.path.join(scores, f"{j}__{c}.json")))
            ["accuracy"]["three_class_accuracy"]
            for c in panel.context_names
        ]
        means.append(float(np.mean(accs)))
    order = list(np.argsort(means)[::-1])
    return {k: sorted(int(x) for x in order[:k]) for k in range(1, panel.N + 1)}


def random_chain(n: int, seed: int) -> Dict[int, List[int]]:
    order = list(np.random.default_rng(seed).permutation(n))
    return {k: sorted(int(x) for x in order[:k]) for k in range(1, n + 1)}


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(
    home: Panel,
    away: Panel,
    home_scores: str,
    budgets: Sequence[int],
    calibration_sizes: Sequence[int],
    seed: int,
) -> Dict[str, object]:
    if home.judge_ids != away.judge_ids:
        raise RuntimeError(
            f"the two benchmarks were scored with different judges: "
            f"{home.judge_ids} vs {away.judge_ids}"
        )

    home_fit = home.splits["FIT"]
    away_fit = away.splits["FIT"]
    n_calib = away.n_items["FIT"]

    # Nested calibration samples, so the efficiency curve compares samples that
    # only ever grow.  One JudgeBench row is one base item, so sampling rows
    # cannot leak an item across the calibration / evaluation boundary.
    order = np.random.default_rng(seed).permutation(n_calib)
    sizes = [m for m in calibration_sizes if m <= n_calib]
    if n_calib not in sizes:
        sizes.append(n_calib)

    methods: Dict[str, Dict[int, List[int]]] = {
        "coverage_backward": backward_chain(home_fit, home.judge_ids),
        "top_accuracy": top_accuracy_chain(home, home_scores),
    }
    for d in range(N_RANDOM_DRAWS):
        methods[f"random_{d}"] = random_chain(home.N, 2000 + d)

    # The diagnostic upper bound: what JudgeBench would have picked for itself.
    oracle = backward_chain(away_fit, away.judge_ids)

    results: Dict[str, object] = {}
    for name, chain in methods.items():
        print(f"[{name}]", flush=True)
        rows: Dict[str, object] = {}
        for k in budgets:
            started = time.time()
            kept = chain[k]

            frozen = fit_basis_weights(home_fit, kept, home.N)
            in_domain = score(home, "TEST", kept, frozen)
            transfer = score(away, "TEST", kept, frozen)

            refit = {}
            for m in sizes:
                sample = np.sort(order[:m])
                w = fit_basis_weights(_rows(away_fit, sample), kept, away.N)
                refit[str(m)] = score(away, "TEST", kept, w)

            oracle_kept = oracle[k]
            oracle_w = fit_basis_weights(away_fit, oracle_kept, away.N)
            oracle_row = score(away, "TEST", oracle_kept, oracle_w)

            rows[str(k)] = {
                "kept": [home.judge_ids[j] for j in kept],
                "in_domain_TEST": in_domain,
                "frozen_weights_TEST": transfer,
                "transfer_gap": (
                    transfer["worst_judge_worst_context_tv"]
                    - in_domain["worst_judge_worst_context_tv"]
                ),
                "refit_weights_TEST": refit,
                "oracle_reselection": {
                    "kept": [away.judge_ids[j] for j in oracle_kept],
                    "TEST": oracle_row,
                    "basis_overlap_jaccard": jaccard(kept, oracle_kept),
                },
                "seconds": time.time() - started,
            }
            best_refit = refit[str(sizes[-1])]["worst_judge_worst_context_tv"]
            print(
                f"  k={k:2d}  in-domain {in_domain['worst_judge_worst_context_tv']:.4f}"
                f"  frozen {transfer['worst_judge_worst_context_tv']:.4f}"
                f"  refit {best_refit:.4f}"
                f"  oracle {oracle_row['worst_judge_worst_context_tv']:.4f}"
                f"  overlap {jaccard(kept, oracle_kept):.2f}"
                f"  ({rows[str(k)]['seconds']:.0f}s)",
                flush=True,
            )
        results[name] = rows

    return {
        "experiment": "E7",
        "claims": ["C2", "C4", "C5"],
        "seed": seed,
        "judges": home.judge_ids,
        "home_benchmark": "allenai/reward-bench-2",
        "away_benchmark": "ScalerLab/JudgeBench",
        "away_has_ties": False,
        "budgets": list(budgets),
        "calibration_sizes": sizes,
        "n_random_draws": N_RANDOM_DRAWS,
        "split_items": {"home": home.n_items, "away": away.n_items},
        "methods": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(PANELS), default="core20")
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--budgets", type=int, nargs="*", default=BUDGETS)
    parser.add_argument("--calibration-sizes", type=int, nargs="*",
                        default=CALIBRATION_SIZES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    home_scores = scores_dir(args.panel)
    away_scores = scores_dir(args.panel, "judgebench")
    if not os.path.isdir(away_scores):
        raise SystemExit(
            f"no JudgeBench blocks at {away_scores}.\n"
            f"Score them first:\n"
            f"  python -u experiments/build_judgebench_pairs.py\n"
            f"  python -u experiments/score_panel.py --gate g2 --panel {args.panel} "
            f"--dataset judgebench --items 620 --scores-only --evict"
        )

    home = load_panel(args.seed, home_scores)
    away = load_panel(args.seed, away_scores, dataset="judgebench")
    print(f"judges: {home.judge_ids}")
    print(f"home items: {home.n_items}")
    print(f"away items: {away.n_items}")

    payload = run(home, away, home_scores, args.budgets,
                  args.calibration_sizes, args.seed)

    out = args.out or os.path.join(ROOT, "results", f"e7_transfer_{args.panel}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
