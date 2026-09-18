"""E5 — certification reliability (plan section 27, claim C7).

C7 says a pruning rule that certifies on an independent split with a
simultaneous upper confidence bound should violate its stated tolerance about
as often as the nominal level allows, and less often than an empirical-only
rule.

The simulation the plan prescribes, one repetition per split:

1. select on FIT   -- greedy backward elimination, so the panel is fixed
                      before CERT is looked at;
2. certify on CERT -- each rule turns the measured losses into one number and
                      accepts the panel when that number is at most gamma;
3. evaluate on TEST -- the panel *violates* when its true worst judge-context
                      error on the locked split exceeds gamma.

The headline is then conditional: among the (split, budget) cases a rule
certified, how often did TEST actually break the promise?  A rule that
certifies nothing has a violation rate of zero and is useless, so the
certified fraction is reported beside it -- the two together are the claim.

Because the response tensor is cached, a repetition costs no inference, and the
split grid is therefore much larger than the five predeclared seeds: those
would resolve a 5% violation rate no better than 0 out of 5.  The five
predeclared seeds lead the list so the primary setting stays inside it.

Rules, matching the plan's comparison list:

`empirical_fit`
    the naive rule: accept when the error measured on the *fitting* items is at
    most gamma.  No independent split and no bound.
`empirical_cert`
    independent split, but the plain measured mean.  Isolates what the split
    buys on its own.
`normal_uncorrected`, `bernstein_uncorrected`
    per-pair bounds with no multiplicity correction.
`bernstein_bonferroni`
    PRIMARY.  Empirical Bernstein at delta / (number of judge-context pairs).
`bootstrap_max`
    grouped bootstrap of the maximum statistic; simultaneous, but pays for the
    dependence actually present rather than the worst case.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.certification import (
    bootstrap_max_error,
    certified_error,
    empirical_error,
    fit_reconstructions,
    pair_samples,
    worst_test_error,
)
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

from experiment_e1_compression_frontier import select_backward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "c7_certification.json")

# Panel sizes to certify.  Below 4 no rule certifies anything at the tolerances
# on the grid, and at 20 the panel reconstructs itself exactly, so neither end
# carries information about the certificate.
K_GRID = [4, 6, 8, 10, 12, 14, 16, 18]

# Tolerances.  Chosen to straddle the errors the panel actually produces (C2
# puts the worst judge at 0.348 for k = 10): at 0.20 almost nothing certifies,
# at 0.60 almost everything does, and the interesting behaviour is between.
GAMMAS = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]

# Primary level first; the plan asks for all three.
DELTAS = [0.05, 0.10, 0.01]

N_BOOT = 1000


def certification_statistics(
    panel: Panel, kept: List[int], deltas: List[float], boot_seed: int
) -> Dict[str, object]:
    """Every rule's number for one (split, budget), plus the TEST truth."""
    kept_set = set(kept)
    weights = fit_reconstructions(panel.splits["FIT"], kept_set, panel.N)

    fit_samples = pair_samples(
        panel.splits["FIT"], panel.item_groups["FIT"], kept_set, weights
    )
    cert_samples = pair_samples(
        panel.splits["CERT"], panel.item_groups["CERT"], kept_set, weights
    )

    # Rules that do not look at delta.
    statistics: Dict[str, object] = {
        "empirical_fit": empirical_error(fit_samples),
        "empirical_cert": empirical_error(cert_samples),
    }
    for delta in deltas:
        tag = f"{delta:g}"
        statistics[f"normal_uncorrected@{tag}"] = certified_error(
            cert_samples, delta, "normal", correct=False
        )
        statistics[f"bernstein_uncorrected@{tag}"] = certified_error(
            cert_samples, delta, "bernstein", correct=False
        )
        statistics[f"bernstein_bonferroni@{tag}"] = certified_error(
            cert_samples, delta, "bernstein", correct=True
        )
        statistics[f"bootstrap_max@{tag}"] = bootstrap_max_error(
            cert_samples, delta, n_boot=N_BOOT, seed=boot_seed
        )

    return {
        "k": len(kept),
        "kept": [panel.judge_ids[j] for j in kept],
        "n_pairs_certified": len(cert_samples),
        "statistics": statistics,
        "test_worst_error": worst_test_error(panel.splits["TEST"], kept_set, weights),
    }


_SHARED: Dict[str, object] = {}


def _init_worker(raw, deltas: List[float], k_grid: List[int]) -> None:
    _SHARED["raw"] = raw
    _SHARED["deltas"] = deltas
    _SHARED["k_grid"] = k_grid


def one_repetition(split_seed: int) -> Dict[str, object]:
    """One FIT/CERT/TEST repetition: select, certify, evaluate.

    Repetitions share nothing but the read-only tensor, so they parallelise
    across processes without any coordination.
    """
    raw = _SHARED["raw"]
    deltas = _SHARED["deltas"]
    k_grid = _SHARED["k_grid"]

    started = time.time()
    panel = split_panel(raw, grouped_split(raw.pairs, seed=split_seed))
    chain = select_backward(panel)
    cases = [
        certification_statistics(panel, chain[k], deltas, boot_seed=split_seed + k)
        for k in k_grid
        if k in chain
    ]
    return {
        "split_seed": split_seed,
        "predeclared": split_seed in SPLIT_SEEDS,
        "n_items": panel.n_items,
        "cases": cases,
        "seconds": time.time() - started,
    }


def run(
    raw,
    split_seeds: List[int],
    deltas: List[float],
    k_grid: List[int],
    workers: int = 1,
) -> Dict[str, object]:
    if workers > 1:
        with Pool(workers, initializer=_init_worker,
                  initargs=(raw, deltas, k_grid)) as pool:
            repetitions = []
            for rep, result in enumerate(
                pool.imap_unordered(one_repetition, split_seeds)
            ):
                repetitions.append(result)
                print(f"[{rep + 1}/{len(split_seeds)}] seed {result['split_seed']} "
                      f"{result['seconds']:.1f}s", flush=True)
        repetitions.sort(key=lambda r: split_seeds.index(r["split_seed"]))
    else:
        _init_worker(raw, deltas, k_grid)
        repetitions = []
        for rep, split_seed in enumerate(split_seeds):
            result = one_repetition(split_seed)
            repetitions.append(result)
            print(f"[{rep + 1}/{len(split_seeds)}] seed {split_seed} "
                  f"{result['seconds']:.1f}s", flush=True)

    return {
        "experiment": "E5",
        "claim": "C7",
        "seed": PRIMARY_SEED,
        "judges": raw.judge_ids,
        "models": raw.model_ids,
        "contexts": raw.context_names,
        "split_seeds": split_seeds,
        "predeclared_split_seeds": SPLIT_SEEDS,
        "k_grid": k_grid,
        "gammas": GAMMAS,
        "deltas": deltas,
        "n_boot": N_BOOT,
        "repetitions": repetitions,
        "summary": summarise(repetitions, deltas),
    }


def summarise(
    repetitions: List[Dict[str, object]], deltas: List[float]
) -> Dict[str, object]:
    """Conditional violation rate and certified fraction, per rule and gamma."""
    rule_names = sorted(repetitions[0]["cases"][0]["statistics"])
    total = sum(len(rep["cases"]) for rep in repetitions)

    rows: List[Dict[str, object]] = []
    for rule in rule_names:
        for gamma in GAMMAS:
            certified = 0
            violated = 0
            for rep in repetitions:
                for case in rep["cases"]:
                    if case["statistics"][rule] <= gamma:
                        certified += 1
                        if case["test_worst_error"] > gamma:
                            violated += 1
            rows.append({
                "rule": rule,
                "gamma": gamma,
                "n_cases": total,
                "n_certified": certified,
                "certified_frac": certified / total if total else 0.0,
                "n_violated": violated,
                "violation_rate": violated / certified if certified else float("nan"),
                # Wilson upper bound, so "0 of 3" is not read as a 0% rate.
                "violation_rate_hi": _wilson_upper(violated, certified),
            })
    return {"rows": rows}


def _wilson_upper(successes: int, trials: int, z: float = 1.96) -> float:
    """Upper end of a 95% Wilson interval; nan when nothing was certified."""
    if trials == 0:
        return float("nan")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = (z / denominator) * np.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    )
    return float(min(1.0, centre + half))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None)
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--repetitions", type=int, default=120,
                        help="number of grouped resplits; the plan asks for >=100")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="repetitions run in parallel; they share nothing")
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(
                ROOT, "results", f"c7_certification_{args.panel}.json"
            )

    # Predeclared seeds first, then deterministic extras drawn from a disjoint
    # range so a rerun with more repetitions extends the list rather than
    # changing the ones already reported.
    extras = [900000 + i for i in range(max(0, args.repetitions - len(SPLIT_SEEDS)))]
    split_seeds = (SPLIT_SEEDS + extras)[: args.repetitions]

    raw = load_raw_panel(args.seed, args.scores)
    print(f"panel: {raw.judge_ids}")
    print(f"repetitions: {len(split_seeds)}  k grid: {K_GRID}  workers: {args.workers}")

    payload = run(raw, split_seeds, DELTAS, K_GRID, args.workers)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
