"""Mandatory backbone analysis (plan section 20).

Section 20 forbids reading irreplaceability off a single returned optimum.  A
judge is only mandatory if it appears in *every* minimum-cardinality panel that
meets the tolerance, and that needs the whole set of optima, not one of them.

The plan states the test as three MILP solves per judge: find the optimal
cardinality k*(gamma), then force z_j = 0 and re-solve to see whether a panel of
size k* still exists, then force z_j = 1 to see whether the judge appears in at
least one optimum.  This experiment answers all three questions at once by
enumerating the optima instead of re-solving.  Enumeration is affordable here
because the panel is small (N = 20) and, on the real judges, the tolerances on
the section 23 grid are only met very close to the full panel, so k* sits near N
and the level below it has few subsets.  Enumeration also removes the one thing
a MILP cannot give: it certifies that the reported optima are *all* of them, so
"appears in every optimum" is a checked statement rather than an inference from
repeated solves.

The downward search is pruned with the monotonicity of the coverage functional.
Growing the retained set can only enlarge the simplex each virtual judge is
fitted over, so E(S) never increases when a judge is added, and therefore every
feasible set of size k is a subset of some feasible set of size k + 1.  Only the
children of the feasible sets found at the previous level need to be scored.
`tests/test_judge_coverage.py` pins that monotonicity, and `--no-prune` scores
every subset at each level for anyone who wants the search repeated without
relying on it.

Judges are classified as in section 20:

  mandatory backbone      in every optimum at this tolerance
  optional representative in some but not all optima
  nonessential            in no optimum

Stability is reported across the five predeclared split seeds, which repartition
the cached responses and cost no inference.

Usage:

    python -u experiments/experiment_backbone.py --panel core20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeCoverageFunctional
from epcrc.panel import PANELS, SCORES, SPLIT_SEEDS, Panel, load_panel, scores_dir
from epcrc.rewardbench import PRIMARY_SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "backbone.json")

# The section 23 grid, extended upward.  On the real panel nothing below 0.13 is
# reachable at any size under the full panel, so the declared grid alone would
# classify every judge as mandatory for a trivial reason -- no subset is feasible
# at all.  Section 23 allows the grid to be extended but not reduced, so the
# looser points are added to give the classification something to separate.
BACKBONE_GAMMAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Slack on the monotonicity pruning.  The minimax weights come from a cutting
# plane solver that stops at a finite tolerance, so E(S) can exceed E(S + {j}) by
# a hair without the functional being non-monotone.  Pruning at exactly gamma
# could then drop a parent whose child is feasible.
_MONOTONE_SLACK = 1e-4

# Upper bound on how many subsets one level may score before the search gives up.
# A level that is cut short leaves k* as a bound rather than a proven optimum,
# and the output says so.
MAX_CANDIDATES = 60000


def _coverage(cov: JudgeCoverageFunctional, subset: FrozenSet[int]) -> float:
    score, _ = cov.compute_coverage(set(subset))
    return float(score)


# Subsets at one level are independent, and section 17.2 asks for exactly this
# parallelism.  Each worker keeps its own functional so the per-subset cache
# still helps within a worker; the panel is sent once at pool start rather than
# with every task.
_WORKER: Dict[str, JudgeCoverageFunctional] = {}


def _worker_init(fit_blocks, eval_blocks, context_names, judge_ids) -> None:
    from epcrc.judge import JudgeResponses
    _WORKER["cov"] = JudgeCoverageFunctional(
        JudgeResponses(fit_blocks, context_names),
        JudgeResponses(eval_blocks, context_names),
        judge_ids,
    )


def _worker_score(subset: FrozenSet[int]):
    return subset, _coverage(_WORKER["cov"], subset)


def enumerate_optima(
    panel: Panel,
    eval_split: str,
    gamma_max: float,
    prune: bool,
    max_candidates: int,
    workers: int = 1,
    verbose: bool = True,
) -> Dict[str, object]:
    """Score subsets level by level downward, keeping the ones that meet gamma_max.

    Returns the feasible sets found at every size together with their coverage,
    plus whether each level was scored in full.  Everything the tolerance-by-
    tolerance classification needs is derived from this one pass, because E(S)
    does not depend on gamma -- only the feasibility test does.
    """
    fit, ev = panel.splits["FIT"], panel.splits[eval_split]
    cov = JudgeCoverageFunctional(fit, ev, panel.judge_ids)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(fit.blocks, ev.blocks, fit.context_names, panel.judge_ids),
        )
    full = frozenset(range(panel.N))

    levels: Dict[int, Dict[FrozenSet[int], float]] = {}
    complete: Dict[int, bool] = {}
    scored = {full: _coverage(cov, full)}
    levels[panel.N] = dict(scored)
    complete[panel.N] = True

    # The expansion frontier is kept at gamma_max plus the solver slack, so a
    # parent that misses the tolerance by a rounding error still gets its
    # children scored.  The classification below tests against gamma exactly.
    cut = gamma_max + _MONOTONE_SLACK
    feasible = {full} if scored[full] <= cut else set()
    stopped_at = None

    for k in range(panel.N - 1, 0, -1):
        if not feasible:
            # Nothing at k + 1 met the tolerance, and by monotonicity nothing
            # smaller can either.  The search is finished, not truncated.
            break

        if prune:
            candidates = {
                frozenset(parent - {j}) for parent in feasible for j in parent
            }
        else:
            candidates = {frozenset(c) for c in combinations(range(panel.N), k)}

        if len(candidates) > max_candidates:
            stopped_at = k
            if verbose:
                print(f"  k={k}: {len(candidates)} candidates exceeds the cap "
                      f"of {max_candidates}; stopping", flush=True)
            break

        start = time.time()
        if pool is None:
            level = {s: _coverage(cov, s) for s in candidates}
        else:
            level = dict(pool.map(_worker_score, sorted(candidates, key=sorted),
                                  chunksize=8))
        levels[k] = level
        complete[k] = True
        feasible = {s for s, e in level.items() if e <= cut}
        if verbose:
            best = min(level.values())
            print(f"  k={k}: scored {len(level)} in {time.time() - start:.1f}s, "
                  f"best {best:.4f}, feasible {len(feasible)}", flush=True)

    if pool is not None:
        pool.shutdown()

    return {
        "levels": levels,
        "complete": complete,
        "stopped_at": stopped_at,
        "smallest_level_scored": min(levels),
    }


def classify(
    panel: Panel, search: Dict[str, object], gamma: float
) -> Dict[str, object]:
    """Apply the section 20 categories at one tolerance.

    `optima` is every minimum-cardinality feasible panel, so the three MILP
    queries in section 20 reduce to membership tests: a judge missing from some
    optimum is exactly a judge for which forcing z_j = 0 still admits a size-k*
    solution, and a judge in some optimum is exactly one for which forcing
    z_j = 1 does.
    """
    levels: Dict[int, Dict[FrozenSet[int], float]] = search["levels"]
    feasible_by_k = {
        k: [s for s, e in level.items() if e <= gamma]
        for k, level in levels.items()
    }
    sizes = sorted(k for k, sets in feasible_by_k.items() if sets)
    if not sizes:
        return {
            "gamma": gamma,
            "k_star": None,
            "feasible": False,
            "certified": True,
            "note": "no subset of any scored size meets this tolerance",
        }

    k_star = min(sizes)
    optima = feasible_by_k[k_star]

    # k* is only the true optimum if the level below it was scored in full and
    # came back empty.  If the search stopped at k* - 1 because of the candidate
    # cap, a smaller panel might exist and k* is an upper bound.
    below = k_star - 1
    certified = below in levels and not feasible_by_k[below]

    counts = {j: 0 for j in range(panel.N)}
    for s in optima:
        for j in s:
            counts[j] += 1

    mandatory, optional, nonessential = [], [], []
    for j, name in enumerate(panel.judge_ids):
        if counts[j] == len(optima):
            mandatory.append(name)
        elif counts[j] == 0:
            nonessential.append(name)
        else:
            optional.append(name)

    return {
        "gamma": gamma,
        "k_star": k_star,
        "feasible": True,
        "certified": bool(certified),
        "n_optima": len(optima),
        "best_coverage": min(levels[k_star][s] for s in optima),
        "mandatory_backbone": mandatory,
        "optional_representative": optional,
        "nonessential": nonessential,
        "appearance_rate": {
            panel.judge_ids[j]: counts[j] / len(optima) for j in range(panel.N)
        },
        "optima": [sorted(panel.judge_ids[j] for j in s) for s in sorted(
            optima, key=lambda s: sorted(s))][:200],
    }


def run_split(
    panel: Panel,
    eval_split: str,
    gammas: Sequence[float],
    prune: bool,
    max_candidates: int,
    workers: int,
) -> Dict[str, object]:
    search = enumerate_optima(
        panel, eval_split, max(gammas), prune, max_candidates, workers
    )
    rows = [classify(panel, search, g) for g in gammas]
    return {
        "smallest_level_scored": search["smallest_level_scored"],
        "stopped_at": search["stopped_at"],
        "subsets_scored": sum(len(v) for v in search["levels"].values()),
        "by_gamma": rows,
    }


def stability(per_seed: Dict[int, Dict[str, object]],
              gammas: Sequence[float],
              judge_ids: List[str]) -> List[Dict[str, object]]:
    """How often each category survives a repartition of the same responses.

    Section 20 asks for this because a category that flips when the split moves
    is a property of the partition, not of the judge.
    """
    out = []
    for i, gamma in enumerate(gammas):
        rows = [seed["by_gamma"][i] for seed in per_seed.values()]
        usable = [r for r in rows if r["feasible"]]
        if not usable:
            out.append({"gamma": gamma, "n_seeds_feasible": 0})
            continue
        labels = {}
        for name in judge_ids:
            tally = {"mandatory": 0, "optional": 0, "nonessential": 0}
            for r in usable:
                if name in r["mandatory_backbone"]:
                    tally["mandatory"] += 1
                elif name in r["optional_representative"]:
                    tally["optional"] += 1
                else:
                    tally["nonessential"] += 1
            labels[name] = tally
        out.append({
            "gamma": gamma,
            "n_seeds_feasible": len(usable),
            "k_star": [r["k_star"] for r in usable],
            "always_mandatory": sorted(
                n for n, t in labels.items() if t["mandatory"] == len(usable)),
            "never_in_any_optimum": sorted(
                n for n, t in labels.items() if t["nonessential"] == len(usable)),
            "unstable": sorted(
                n for n, t in labels.items()
                if max(t.values()) < len(usable)),
            "counts": labels,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None)
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--eval-split", default="FIT", choices=["FIT", "CERT", "TEST"],
                        help="split the coverage is scored on; the optimum is "
                             "defined on the design matrix, so FIT is the default")
    parser.add_argument("--split-seeds", type=int, nargs="*", default=SPLIT_SEEDS)
    parser.add_argument("--gammas", type=float, nargs="*", default=BACKBONE_GAMMAS)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="processes used to score one level; subsets at a "
                             "level are independent")
    parser.add_argument("--no-prune", action="store_true",
                        help="score every subset at each level instead of only "
                             "the children of feasible ones")
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(ROOT, "results", f"backbone_{args.panel}.json")

    gammas = sorted(args.gammas)
    per_seed: Dict[int, Dict[str, object]] = {}
    judge_ids: List[str] = []

    for split_seed in args.split_seeds:
        panel = load_panel(args.seed, args.scores, split_seed=split_seed)
        judge_ids = panel.judge_ids
        print(f"[split seed {split_seed}] N={panel.N} "
              f"items={panel.n_items}", flush=True)
        per_seed[split_seed] = run_split(
            panel, args.eval_split, gammas, not args.no_prune,
            args.max_candidates, args.workers,
        )
        for row in per_seed[split_seed]["by_gamma"]:
            if not row["feasible"]:
                print(f"  gamma={row['gamma']}: infeasible at every scored size",
                      flush=True)
            else:
                print(f"  gamma={row['gamma']}: k*={row['k_star']} "
                      f"({'certified' if row['certified'] else 'upper bound'}), "
                      f"{row['n_optima']} optima, "
                      f"{len(row['mandatory_backbone'])} mandatory, "
                      f"{len(row['nonessential'])} nonessential", flush=True)

    payload = {
        "experiment": "backbone",
        "plan_section": "20",
        "claim": "C4",
        "seed": args.seed,
        "eval_split": args.eval_split,
        "judges": judge_ids,
        "gammas": gammas,
        "pruned_by_monotonicity": not args.no_prune,
        "max_candidates": args.max_candidates,
        "workers": args.workers,
        "split_seeds": list(args.split_seeds),
        "per_split_seed": {str(k): v for k, v in per_seed.items()},
        "stability": stability(per_seed, gammas, judge_ids),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
