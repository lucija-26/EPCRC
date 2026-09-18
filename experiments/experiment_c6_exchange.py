"""E4 -- exact optimality and exchange structure (plan section 26, claim C6).

C6 says greedy selection gets trapped by set dependence, that a low-order
exchange move (2-swap, then trimming) closes most of the gap to the true
optimum, and that 3-swap adds little for its cost.

Testing that needs a *certified* optimum, so the instances are deliberately
small: ten fixed subpanels of Core-20 with 12 to 16 judges each.  On each
subpanel and each tolerance gamma the experiment reports

    opt(gamma)   the smallest |S| with E(S) <= gamma, proved by enumeration,
    |S| for every greedy and exchange method,
    the extra-model gap between the two.

`E(S)` is the worst-judge worst-context mean TV error of reconstructing every
judge outside S from S, fitted on FIT and scored on CERT.  It does not depend on
gamma, so one cache serves the whole gamma sweep.

**Why the enumeration terminates.**  Three things keep it small.  A set is
infeasible as soon as *one* judge exceeds gamma, so most subsets die after a
couple of solves.  Backward elimination supplies a feasible set, so the optimum
is known to lie at or below its size and sizes above that are never visited.
And gammas are swept downward, so a subset already proved infeasible at a loose
gamma is infeasible at every tighter one and is never re-evaluated.

**Why these gammas.**  The section 23 grid runs from 0.02 to 0.20 and is vacuous
on real judges -- C2 reports a floor of about 0.13 even at k = 19, so no subset
of a 12-to-16 judge subpanel meets any of those points and every instance would
be unsolved.  E4 therefore uses tolerance points inside the range the real panel
actually reaches.  This is a deviation from the plan and is recorded as one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeCoverageFunctional, JudgeResponses, solve_minimax_weights
from epcrc.judge import worst_context_error
from epcrc.panel import PANELS, SCORES, SPLIT_SEEDS, Panel, load_raw_panel, scores_dir
from epcrc.pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)
from epcrc.rewardbench import PRIMARY_SEED, grouped_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "c6_exchange.json")

# Inside the range the real panel reaches; see the module docstring.
GAMMAS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

# Ten fixed subpanels, two at each size from 12 to 16.
SUBPANEL_SIZES = [12, 12, 13, 13, 14, 14, 15, 15, 16, 16]
SUBPANEL_SEED = 606

METHODS = [
    ("forward", lambda cov, g: ForwardSelectionPruner(cov, g, cleanup=False)),
    ("backward", lambda cov, g: BackwardEliminationPruner(cov, g)),
    ("forward_trim", lambda cov, g: ForwardSelectionPruner(cov, g, cleanup=True)),
    ("swap2", lambda cov, g: BackwardKSwapPruner(cov, g, max_swap_k=2)),
    ("swap2_forward", lambda cov, g: ForwardKSwapPruner(cov, g, max_swap_k=2)),
    ("swap2_pq", lambda cov, g: PriorityQueuePruner(cov, g, max_swap_k=2)),
    ("swap3", lambda cov, g: BackwardKSwapPruner(cov, g, max_swap_k=3)),
]


def subpanels(n_judges: int, sizes: Sequence[int]) -> List[List[int]]:
    """The fixed subpanels, drawn once from a fixed seed."""
    rng = np.random.default_rng(SUBPANEL_SEED)
    return [
        sorted(int(i) for i in rng.choice(n_judges, size=size, replace=False))
        for size in sizes
    ]


def restrict(responses: JudgeResponses, judges: Sequence[int]) -> JudgeResponses:
    return JudgeResponses(
        [block[:, list(judges), :] for block in responses.blocks],
        responses.context_names,
    )


class ExactSearch:
    """Smallest feasible |S| by enumeration, with a cache shared across gammas.

    Each subset is stored either as an exact E(S) or as a lower bound left by an
    early exit.  Sweeping gamma downward makes every stored lower bound
    conclusive for all later gammas, so no subset is ever evaluated twice.
    """

    def __init__(self, fit: JudgeResponses, cert: JudgeResponses):
        self.fit = fit
        self.cert = cert
        self.N = fit.n_judges
        self._cache: Dict[FrozenSet[int], Tuple[str, float]] = {}
        self.n_eval = 0
        self.n_hit = 0

    def _feasible(self, combo: Tuple[int, ...], gamma: float) -> Tuple[bool, Optional[float]]:
        key = frozenset(combo)
        cached = self._cache.get(key)
        if cached is not None:
            kind, value = cached
            if kind == "exact":
                self.n_hit += 1
                return value <= gamma, (value if value <= gamma else None)
            if value >= gamma:
                self.n_hit += 1
                return False, None

        self.n_eval += 1
        kept = list(combo)
        kept_set = set(combo)
        worst = 0.0
        for target in range(self.N):
            if target in kept_set:
                continue
            _, w = solve_minimax_weights(self.fit, target, kept)
            u = worst_context_error(self.cert, target, kept, w)
            if u > gamma:
                self._cache[key] = ("lb", u)
                return False, None
            worst = max(worst, u)
        self._cache[key] = ("exact", worst)
        return True, worst

    def min_feasible_size(
        self, gamma: float, upper_bound: int
    ) -> Tuple[int, Optional[Tuple[int, ...]], Optional[float]]:
        """The proved optimum, searching sizes 1 .. upper_bound - 1 first.

        `upper_bound` is the size of a set already known to be feasible, so if
        nothing smaller works the optimum is exactly `upper_bound`.
        """
        for k in range(1, upper_bound):
            for combo in combinations(range(self.N), k):
                ok, value = self._feasible(combo, gamma)
                if ok:
                    return k, combo, value
        return upper_bound, None, None


def run_subpanel(
    panel: Panel, judges: Sequence[int], gammas: List[float]
) -> Dict[str, object]:
    fit = restrict(panel.splits["FIT"], judges)
    cert = restrict(panel.splits["CERT"], judges)
    names = [panel.judge_ids[j] for j in judges]

    cov = JudgeCoverageFunctional(fit, cert, names)
    search = ExactSearch(fit, cert)

    rows: List[Dict[str, object]] = []
    for gamma in sorted(gammas, reverse=True):
        sizes: Dict[str, int] = {}
        kept: Dict[str, List[str]] = {}
        seconds: Dict[str, float] = {}
        for method, factory in METHODS:
            t0 = time.time()
            result = factory(cov, gamma).run()
            seconds[method] = round(time.time() - t0, 3)
            sizes[method] = len(result.kept_set)
            kept[method] = [names[i] for i in sorted(result.kept_set)]

        # Every method returns a feasible set, so the smallest bounds the search.
        best_method = min(sizes, key=lambda m: sizes[m])
        upper_bound = sizes[best_method]

        t0 = time.time()
        opt_size, opt_combo, opt_error = search.min_feasible_size(gamma, upper_bound)
        search_seconds = round(time.time() - t0, 3)

        if opt_combo is None:
            opt_kept = kept[best_method]
            opt_error = float(cov.compute_coverage(
                {names.index(x) for x in opt_kept}
            )[0])
        else:
            opt_kept = [names[i] for i in opt_combo]

        rows.append({
            "gamma": gamma,
            "opt_size": opt_size,
            "opt_error": float(opt_error),
            "opt_kept": opt_kept,
            "sizes": sizes,
            "kept": kept,
            "gap": {m: sizes[m] - opt_size for m in sizes},
            "exact": {m: sizes[m] == opt_size for m in sizes},
            "method_seconds": seconds,
            "search_seconds": search_seconds,
        })
        print(
            f"    gamma {gamma:.2f}  opt {opt_size}  "
            + "  ".join(f"{m} {sizes[m]}" for m, _ in METHODS)
            + f"  ({search_seconds:.1f}s, {search.n_eval} evals)",
            flush=True,
        )

    return {
        "judges": names,
        "n_judges": len(names),
        "subset_evaluations": search.n_eval,
        "cache_hits": search.n_hit,
        "cache_hit_rate": (
            search.n_hit / (search.n_hit + search.n_eval)
            if search.n_hit + search.n_eval else 0.0
        ),
        "rows": rows,
    }


_SHARED: Dict[str, object] = {}


def _init_worker(panel: Panel, gammas: List[float]) -> None:
    _SHARED["panel"] = panel
    _SHARED["gammas"] = gammas


def _one(args: Tuple[int, List[int]]) -> Tuple[int, Dict[str, object]]:
    index, judges = args
    panel = _SHARED["panel"]
    print(f"  [subpanel {index}] n={len(judges)}", flush=True)
    return index, run_subpanel(panel, judges, _SHARED["gammas"])


def summarise(instances: List[Dict[str, object]]) -> Dict[str, object]:
    """Per method: how often it hit the optimum, and by how much it missed."""
    method_names = [m for m, _ in METHODS]
    per_method: Dict[str, Dict[str, float]] = {}
    for method in method_names:
        gaps = np.array(
            [row["gap"][method] for inst in instances for row in inst["rows"]],
            dtype=float,
        )
        seconds = np.array(
            [row["method_seconds"][method] for inst in instances for row in inst["rows"]],
            dtype=float,
        )
        per_method[method] = {
            "n_instances": int(gaps.size),
            "exact_rate": float((gaps == 0).mean()),
            "mean_gap": float(gaps.mean()),
            "max_gap": float(gaps.max()),
            "mean_seconds": float(seconds.mean()),
            "total_seconds": float(seconds.sum()),
        }

    swap2 = np.array(
        [row["gap"]["swap2"] for inst in instances for row in inst["rows"]], dtype=float
    )
    swap3 = np.array(
        [row["gap"]["swap3"] for inst in instances for row in inst["rows"]], dtype=float
    )
    t2 = per_method["swap2"]["total_seconds"]
    t3 = per_method["swap3"]["total_seconds"]

    return {
        "per_method": per_method,
        "swap3_over_swap2": {
            "mean_gap_reduction": float((swap2 - swap3).mean()),
            "n_instances_improved": int((swap3 < swap2).sum()),
            "n_instances_worse": int((swap3 > swap2).sum()),
            "time_ratio": float(t3 / t2) if t2 else float("nan"),
        },
        "target_section_26": {
            "swap2_exact_rate_at_least_0.70": per_method["swap2"]["exact_rate"] >= 0.70,
            "swap2_mean_gap_at_most_0.5": per_method["swap2"]["mean_gap"] <= 0.5,
            "swap2_max_gap_at_most_1": per_method["swap2"]["max_gap"] <= 1.0,
        },
    }


def run(
    raw, split_seed: int, gammas: List[float], workers: int, sizes: Sequence[int]
) -> Dict[str, object]:
    from epcrc.panel import split_panel

    panel = split_panel(raw, grouped_split(raw.pairs, seed=split_seed))
    sets = subpanels(panel.N, sizes)
    print(f"subpanel sizes: {[len(s) for s in sets]}", flush=True)

    jobs = list(enumerate(sets))
    instances: List[Optional[Dict[str, object]]] = [None] * len(jobs)
    if workers > 1:
        with Pool(workers, initializer=_init_worker, initargs=(panel, gammas)) as pool:
            for index, result in pool.imap_unordered(_one, jobs):
                instances[index] = result
    else:
        _init_worker(panel, gammas)
        for job in jobs:
            index, result = _one(job)
            instances[index] = result

    done = [inst for inst in instances if inst is not None]
    return {
        "experiment": "E4",
        "claim": "C6",
        "seed": PRIMARY_SEED,
        "split_seed": split_seed,
        "judges": raw.judge_ids,
        "contexts": raw.context_names,
        "gammas": sorted(gammas),
        "subpanel_sizes": list(sizes),
        "subpanel_seed": SUBPANEL_SEED,
        "methods": [m for m, _ in METHODS],
        "note_gamma_grid": (
            "The section 23 grid (0.02 to 0.20) is vacuous on real judges: C2 "
            "measures a floor near 0.13 at k = 19, so no subpanel of 12 to 16 "
            "judges reaches any of those points and every instance would be "
            "unsolved. E4 uses tolerance points inside the achievable range. "
            "Recorded as a deviation from the plan."
        ),
        "note_fit_and_score": (
            "E(S) is fitted on FIT and scored on CERT, so the optimum being "
            "certified is a held-out optimum rather than a training fit."
        ),
        "summary": summarise(done),
        "instances": done,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None)
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEEDS[0])
    parser.add_argument("--gammas", type=float, nargs="*", default=GAMMAS)
    parser.add_argument("--subpanels", type=int, default=len(SUBPANEL_SIZES),
                        help="use only the first n subpanels (smoke tests)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(ROOT, "results", f"c6_exchange_{args.panel}.json")

    sizes = SUBPANEL_SIZES[: args.subpanels]

    raw = load_raw_panel(args.seed, args.scores)
    print(f"panel: {raw.judge_ids}")
    print(f"gammas: {args.gammas}  workers: {args.workers}")

    started = time.time()
    payload = run(raw, args.split_seed, args.gammas, args.workers, sizes)
    payload["wall_seconds"] = round(time.time() - started, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("\nmethod          exact%   mean gap   max gap   mean s")
    for method, stats in payload["summary"]["per_method"].items():
        print(
            f"{method:14s} {100 * stats['exact_rate']:6.1f}   "
            f"{stats['mean_gap']:8.2f}  {stats['max_gap']:8.0f}  "
            f"{stats['mean_seconds']:7.2f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
