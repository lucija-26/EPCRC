"""Experiment E8 -- synthetic structure and sample complexity (plan section 30).

E8 is the one experiment where the right answer is known before the code runs.
`archetype_panel` plants a backbone of archetypal judges and builds every other
judge as an exact convex combination of them, which makes the planted set the
unique minimum panel at zero tolerance (the generator's docstring proves it).
So each quantity below has a ground truth to be checked against, rather than
only a plausible-looking value:

  1. backbone recovery      -- does selection return the planted archetypes?
  2. non-composability      -- do individual certificates fail jointly, where
                               the construction says they must?
  3. exact-vs-heuristic gap -- backward elimination against exhaustive search
                               at equal panel size.
  4. certification calibration -- a tolerance met on CERT, tested on TEST.
  5. query scaling          -- how many items recovery needs.

The sweep is one-factor-at-a-time around a base configuration, over several
seeds.  Full factorial over seven axes is not affordable and would not answer
anything the marginals do not, because the axes are not being tested for
interaction.

Plan section 30 also lists "intrinsic response dimension" as an axis.  It is
not swept separately because it is not free: stacked over items, the judge
cloud spans the affine hull of the archetypes, so its intrinsic dimension is
exactly `n_archetypes - 1`.  The archetype axis is the dimension axis.

This experiment is pure numpy -- no judge inference -- so it runs anywhere and
needs no scored panel.

Usage:

    python -u experiments/experiment_e8_synthetic.py
    python -u experiments/experiment_e8_synthetic.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeCoverageFunctional
from epcrc.synthetic_judges import ArchetypePanel, archetype_panel
from experiments.experiment_e0_noncomposability import (
    dependency_cycles,
    greedy_feasible_panel,
    leave_one_out,
)

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
OUT = os.path.join(RESULTS, "e8_synthetic.json")

SEEDS = [0, 1, 2, 3, 4]

# Base configuration.  Small enough that exhaustive search over subsets of the
# backbone's size stays exact on every axis, which is what item 3 needs.
BASE = {
    "n_judges": 10,
    "n_archetypes": 4,
    "n_items": 200,
    "n_contexts": 2,
    "noise": 0.02,
    "noise_correlation": 0.5,
    "n_duplicates": 2,
}

AXES = {
    "n_judges": [8, 10, 12, 14],
    "n_archetypes": [3, 4, 5, 6],
    "n_items": [25, 50, 100, 200, 400],
    "n_contexts": [1, 2, 4],
    "noise": [0.0, 0.01, 0.02, 0.05, 0.10],
    "noise_correlation": [0.0, 0.5, 1.0],
    "n_duplicates": [0, 2, 4],
}

# Tolerances for the calibration and non-composability items.  Chosen to
# straddle the reconstruction error the base noise level produces.
GAMMAS = [0.01, 0.02, 0.04, 0.08, 0.16]

DEPENDENCY_TAU = 0.05


def exhaustive_best(
    cov: JudgeCoverageFunctional, k: int
) -> Tuple[List[int], float]:
    """The size-k panel with the smallest coverage error, by full enumeration.

    Exact by construction, and affordable here only because E8 keeps N small on
    purpose.  This is the reference the greedy answer is scored against.
    """
    best_subset, best_error = None, float("inf")
    for combo in combinations(range(cov.N), k):
        error, _ = cov.compute_coverage(set(combo))
        if error < best_error:
            best_subset, best_error = list(combo), float(error)
    return best_subset, best_error


def backward_to_size(cov: JudgeCoverageFunctional, k: int) -> Tuple[List[int], float]:
    """Backward elimination run down to exactly k judges, ignoring tolerance.

    Fixing the size is what makes the greedy and exhaustive answers comparable:
    both are then asked the same question, and the difference between their
    coverage errors is the heuristic's whole cost.
    """
    current = set(range(cov.N))
    while len(current) > k:
        _, drop = min(
            ((cov.compute_coverage(current - {j})[0], j) for j in sorted(current)),
            key=lambda candidate: candidate[0],
        )
        current = current - {drop}
    kept = sorted(current)
    return kept, float(cov.compute_coverage(set(kept))[0])


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def recovery(panel: ArchetypePanel, cov: JudgeCoverageFunctional) -> Dict[str, object]:
    """Items 1 and 3: does selection at the true size return the true backbone?

    Both selectors are asked for exactly `len(backbone)` judges, so a miss is a
    selection error and never a size mismatch.  The exhaustive answer doubles as
    the optimality reference for the greedy one.
    """
    k = len(panel.backbone)
    greedy_set, greedy_error = backward_to_size(cov, k)
    exact_set, exact_error = exhaustive_best(cov, k)
    truth_error, _ = cov.compute_coverage(set(panel.backbone))

    return {
        "backbone": list(panel.backbone),
        "backbone_coverage": float(truth_error),
        "greedy_set": greedy_set,
        "greedy_coverage": greedy_error,
        "greedy_recovers_backbone": greedy_set == list(panel.backbone),
        "greedy_jaccard": jaccard(greedy_set, panel.backbone),
        "exact_set": exact_set,
        "exact_coverage": exact_error,
        "exact_recovers_backbone": exact_set == list(panel.backbone),
        "exact_jaccard": jaccard(exact_set, panel.backbone),
        # Positive means greedy paid for being greedy.  It cannot be negative:
        # the exhaustive answer is the minimum over all size-k panels.
        "greedy_gap": greedy_error - exact_error,
        "greedy_is_exact": bool(np.isclose(greedy_error, exact_error, atol=1e-9)),
    }


def composition(
    panel: ArchetypePanel, cov: JudgeCoverageFunctional, gammas: Sequence[float]
) -> List[Dict[str, object]]:
    """Item 2: the section 22 audit, on a panel whose dependencies were planted.

    The duplicate pairs are the planted cycles -- each copy is reconstructed
    exactly by its original, so both look individually removable and neither
    can go once the other has.
    """
    loo_errors, loo_weights = leave_one_out(cov)
    graph = dependency_cycles(loo_weights, DEPENDENCY_TAU)

    rows = []
    for gamma in gammas:
        removable = [i for i in range(cov.N) if loo_errors[i] <= gamma]
        naive = [i for i in range(cov.N) if i not in set(removable)]
        naive_error, _ = cov.compute_coverage(set(naive))
        rows.append({
            "gamma": gamma,
            "n_individually_removable": len(removable),
            "naive_retained_size": len(naive),
            "naive_coverage": float(naive_error),
            "naive_violates_gamma": bool(naive_error > gamma),
            "n_cycles": graph["n_cycles"],
            "largest_cycle": graph["largest_cycle"],
            "n_planted_duplicates": len(panel.duplicate_of),
        })
    return rows


def calibration(
    panel: ArchetypePanel, gammas: Sequence[float]
) -> List[Dict[str, object]]:
    """Item 4: a panel certified on CERT, then checked on held-out TEST.

    Weights are fitted on FIT and the tolerance is verified on CERT, exactly as
    the real pipeline does it; TEST is touched only to ask whether the
    certificate held.  Each split is a fresh draw of items from the same judges,
    so a violation is sampling error and nothing else.
    """
    fit, cert, test = panel.splits["FIT"], panel.splits["CERT"], panel.splits["TEST"]
    on_cert = JudgeCoverageFunctional(fit, cert, panel.names)
    on_test = JudgeCoverageFunctional(fit, test, panel.names)

    rows = []
    for gamma in gammas:
        _, kept, cert_error = greedy_feasible_panel(on_cert, gamma)
        test_error, _ = on_test.compute_coverage(set(kept))
        rows.append({
            "gamma": gamma,
            "selected_size": len(kept),
            "cert_coverage": float(cert_error),
            "test_coverage": float(test_error),
            "cert_within_gamma": bool(cert_error <= gamma),
            "test_violates_gamma": bool(test_error > gamma),
            "test_minus_cert": float(test_error - cert_error),
        })
    return rows


def run_config(
    axis: str, value: object, config: Dict[str, object], seed: int
) -> Dict[str, object]:
    panel = archetype_panel(seed=seed, **config)
    cov = JudgeCoverageFunctional(
        panel.splits["FIT"], panel.splits["CERT"], panel.names
    )
    return {
        "axis": axis,
        "value": value,
        "config": dict(config),
        "seed": seed,
        "recovery": recovery(panel, cov),
        "composition": composition(panel, cov, GAMMAS),
        "calibration": calibration(panel, GAMMAS),
    }


def sweep(seeds: Sequence[int], axes: Dict[str, List]) -> List[Dict[str, object]]:
    """One factor at a time around BASE, skipping repeats of the base cell."""
    records: List[Dict[str, object]] = []
    seen_configs = set()

    for axis, values in axes.items():
        for value in values:
            config = dict(BASE)
            config[axis] = value
            # n_duplicates is bounded by how many interior judges exist, so some
            # cells of the judge/archetype axes are simply not constructible.
            interior = config["n_judges"] - config["n_archetypes"]
            if config["n_duplicates"] > max(0, interior - 1):
                continue

            key = tuple(sorted(config.items()))
            if key in seen_configs:
                continue
            seen_configs.add(key)

            started = time.time()
            for seed in seeds:
                records.append(run_config(axis, value, config, seed))
            print(f"  {axis}={value}  {time.time() - started:.1f}s", flush=True)

    return records


def _mean(rows: Sequence[Dict], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        values.append(float(node))
    return float(np.mean(values)) if values else float("nan")


def summarise(records: List[Dict[str, object]]) -> None:
    print("\n--- 1 & 3: backbone recovery and the greedy gap "
          "(at the true panel size) ---")
    header = (f"{'axis':<18}{'value':>8}{'exact_rec':>11}{'greedy_rec':>12}"
              f"{'greedy_gap':>12}{'E(true)':>10}")
    print(header)
    print("-" * len(header))
    for axis in AXES:
        rows_axis = [r for r in records if r["axis"] == axis]
        for value in sorted({str(r["value"]) for r in rows_axis}):
            group = [r for r in rows_axis if str(r["value"]) == value]
            print(
                f"{axis:<18}{value:>8}"
                f"{_mean(group, ('recovery', 'exact_recovers_backbone')):>11.2f}"
                f"{_mean(group, ('recovery', 'greedy_recovers_backbone')):>12.2f}"
                f"{_mean(group, ('recovery', 'greedy_gap')):>12.4f}"
                f"{_mean(group, ('recovery', 'backbone_coverage')):>10.4f}"
            )

    print("\n--- 2: non-composability, pooled over the sweep ---")
    header = (f"{'gamma':>8}{'removable':>11}{'S_naive':>9}{'E(S_naive)':>12}"
              f"{'violates':>10}{'cycles':>8}")
    print(header)
    print("-" * len(header))
    for gamma in GAMMAS:
        rows = [c for r in records for c in r["composition"] if c["gamma"] == gamma]
        print(
            f"{gamma:>8.3f}{_mean(rows, ('n_individually_removable',)):>11.2f}"
            f"{_mean(rows, ('naive_retained_size',)):>9.2f}"
            f"{_mean(rows, ('naive_coverage',)):>12.4f}"
            f"{_mean(rows, ('naive_violates_gamma',)):>10.2f}"
            f"{_mean(rows, ('n_cycles',)):>8.2f}"
        )

    print("\n--- 4: certification calibration (CERT-selected, TEST-checked) ---")
    header = (f"{'gamma':>8}{'|S|':>7}{'E(CERT)':>10}{'E(TEST)':>10}"
              f"{'TEST-CERT':>11}{'violation':>11}")
    print(header)
    print("-" * len(header))
    for gamma in GAMMAS:
        rows = [c for r in records for c in r["calibration"] if c["gamma"] == gamma]
        print(
            f"{gamma:>8.3f}{_mean(rows, ('selected_size',)):>7.2f}"
            f"{_mean(rows, ('cert_coverage',)):>10.4f}"
            f"{_mean(rows, ('test_coverage',)):>10.4f}"
            f"{_mean(rows, ('test_minus_cert',)):>11.4f}"
            f"{_mean(rows, ('test_violates_gamma',)):>11.2f}"
        )

    print("\n--- 5: query scaling (items axis) ---")
    header = (f"{'n_items':>9}{'exact_rec':>11}{'greedy_rec':>12}"
              f"{'E(true)':>10}{'TEST-CERT':>11}")
    print(header)
    print("-" * len(header))
    for value in AXES["n_items"]:
        group = [r for r in records
                 if r["axis"] == "n_items" and r["value"] == value]
        if not group:
            continue
        drift = [c for r in group for c in r["calibration"]]
        print(
            f"{value:>9}"
            f"{_mean(group, ('recovery', 'exact_recovers_backbone')):>11.2f}"
            f"{_mean(group, ('recovery', 'greedy_recovers_backbone')):>12.2f}"
            f"{_mean(group, ('recovery', 'backbone_coverage')):>10.4f}"
            f"{_mean(drift, ('test_minus_cert',)):>11.4f}"
        )


def main(quick: bool = False) -> None:
    os.makedirs(RESULTS, exist_ok=True)

    seeds = SEEDS[:2] if quick else SEEDS
    axes = {"noise": AXES["noise"]} if quick else AXES

    started = time.time()
    records = sweep(seeds, axes)
    elapsed = time.time() - started

    with open(OUT, "w") as handle:
        json.dump({
            "experiment": "E8",
            "claims": ["C1", "C6", "C7"],
            "base_config": BASE,
            "axes": axes,
            "gammas": GAMMAS,
            "seeds": list(seeds),
            "elapsed_seconds": elapsed,
            "records": records,
        }, handle, indent=2)

    print(f"\nwrote {OUT}  ({elapsed:.1f}s, {len(records)} runs)")
    summarise(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="two seeds and the noise axis only, for a smoke test")
    args = parser.parse_args()
    main(args.quick)
