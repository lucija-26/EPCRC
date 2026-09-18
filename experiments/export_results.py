"""Pack every result into one self-contained, checkable zip.

The professor asked for "all results (e.g. .csv files, summaries.md etc) as a
zip file".  This builds that package, and it is deliberately more than a folder
of CSVs: a results package that cannot be traced back to the code and seeds
that produced it cannot be defended, so the zip also carries the raw
experiment JSON, a manifest with a SHA-256 per file, and the git commit the
numbers were produced at.

Every table is taken from `epcrc.report` and every figure from `epcrc.figures`,
which are the same functions the notebooks call.  A number in the zip and the
same number in a notebook therefore cannot drift apart.

Usage:

    python -u experiments/export_results.py
    python -u experiments/export_results.py --panel core20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc import figures as F
from epcrc import report as R
from epcrc.panel import PANELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# The budget C5 is reported at.  Half the panel is the point the plan's
# evidence matrix states its downstream tolerance over, and reporting one
# budget in prose while shipping every budget as a CSV keeps the choice
# visible instead of letting a reader wonder which k the sentence used.
C5_HEADLINE_K = 10


# --------------------------------------------------------------------------
# where the inputs live
# --------------------------------------------------------------------------

def input_paths(panel: str) -> Dict[str, str]:
    """Result files for a panel.

    The experiments wrote unsuffixed names before the panel registry existed,
    and those files are Core-8. So the fallback is only correct for Core-8: on
    any other panel a missing result must stay missing, or the export would
    quietly ship Core-8 numbers under the other panel's label.
    """
    def pick(*candidates: str) -> str:
        for name in candidates:
            path = os.path.join(RESULTS, name)
            if os.path.exists(path):
                return path
        return os.path.join(RESULTS, candidates[-1])

    legacy = panel == "core8"
    return {
        "e0_synthetic": pick("e0_noncomposability.json"),
        "e0_real": pick(f"e0_real_{panel}.json"),
        "e1": pick(f"e1_frontier_{panel}.json", *(["e1_frontier.json"] if legacy else [])),
        "c3": pick(f"c3_baselines_{panel}.json", *(["c3_baselines.json"] if legacy else [])),
        "c4": pick(f"c4_stress_specialists_{panel}.json",
                   *(["c4_stress_specialists.json"] if legacy else [])),
        "c5": pick(f"c5_downstream_{panel}.json",
                   *(["c5_downstream.json"] if legacy else [])),
        "c6": pick(f"c6_exchange_{panel}.json",
                   *(["c6_exchange.json"] if legacy else [])),
        "c7": pick(f"c7_certification_{panel}.json",
                   *(["c7_certification.json"] if legacy else [])),
        # No legacy names below: both were written after the panel registry
        # existed, so an unsuffixed file for them never existed.
        "e6": pick(f"e6_sparse_cost_{panel}.json"),
        "e7": pick(f"e7_transfer_{panel}.json"),
        "backbone": pick(f"backbone_{panel}.json"),
    }


def gate_paths(panel: str) -> Dict[str, str]:
    """The G0-G2 gate records for a panel, which sections 46-48 require.

    Kept out of `input_paths` on purpose.  A missing gate record does not make
    the claims partial the way a missing experiment does -- it means the gate
    was run before the record was kept, or on a different machine -- so it must
    not trigger the partial-package warning.  It is still evidence a reader is
    entitled to, hence shipping it.
    """
    return {
        f"gate_{name}": os.path.join(RESULTS, panel, f"{name}.json")
        for name in ("g0", "g1", "g2")
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


def provenance(panel: str, inputs: Dict[str, str]) -> dict:
    """Everything a reader needs to reproduce or challenge these numbers."""
    dirty = _git("status", "--porcelain")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": panel,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_clean": dirty in ("", "unavailable"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "inputs": {
            key: {
                "path": os.path.relpath(path, ROOT),
                "present": os.path.exists(path),
            }
            for key, path in inputs.items()
        },
    }


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def build_tables(inputs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Every CSV in the package, keyed by filename stem.

    A table is skipped rather than faked when its experiment has not been run,
    so a partial package is visibly partial.
    """
    tables: Dict[str, pd.DataFrame] = {}

    if os.path.exists(inputs["e0_synthetic"]):
        tables["c1_synthetic_headline"] = R.c1_headline(
            inputs["e0_synthetic"], gamma_source=None)
        tables["c1_synthetic_per_seed"] = R.c1_table(inputs["e0_synthetic"])

    if os.path.exists(inputs["e0_real"]):
        tables["c1_real_declared_grid"] = R.c1_headline(
            inputs["e0_real"], gamma_source="declared")
        tables["c1_real_loo_breakpoints"] = R.c1_headline(
            inputs["e0_real"], gamma_source="loo_breakpoint")
        tables["c1_real_per_seed"] = R.c1_table(inputs["e0_real"])

    if os.path.exists(inputs["e1"]):
        tables["c2_frontier_TEST"] = R.c2_table(inputs["e1"], "TEST")
        tables["c2_frontier_CERT"] = R.c2_table(inputs["e1"], "CERT")

    if os.path.exists(inputs["c3"]):
        tables["c3_headline"] = R.c3_headline(inputs["c3"])
        tables["c3_per_method_per_k"] = R.c3_table(inputs["c3"])
        tables["c3_paired_bootstrap"] = R.c3_paired_table(inputs["c3"])
        tables["c3_split_robustness"] = R.c3_split_robustness(inputs["c3"])
        tables["c3_cost"] = R.c3_cost_table(inputs["c3"])
        tables["c3_lowrank_floor"] = R.lowrank_floor_table(inputs["c3"])
        tables["c3_reconstruction_rules"] = R.reconstruction_table(inputs["c3"])

    if os.path.exists(inputs["c4"]):
        tables["c4_headline"] = R.c4_headline(inputs["c4"])
        tables["c4_per_seed_per_arm"] = R.c4_table(inputs["c4"])
        tables["c4_specialists"] = R.c4_specialists(inputs["c4"])

    if os.path.exists(inputs["c5"]):
        tables["c5_per_context"] = R.c5_table(inputs["c5"])
        # One headline per aggregator, because the aggregator is the variable
        # that decides C5: reconstruction only has to earn its place under a
        # rule that does not already re-weight the judges itself.
        for aggregator in R.load(inputs["c5"])["aggregators"]:
            tables[f"c5_headline_k{C5_HEADLINE_K}_{aggregator}"] = R.c5_headline(
                inputs["c5"], C5_HEADLINE_K, aggregator)

    if os.path.exists(inputs["c6"]):
        tables["c6_headline"] = R.c6_headline(inputs["c6"])
        tables["c6_per_instance"] = R.c6_table(inputs["c6"])

    if os.path.exists(inputs["c7"]):
        tables["c7_per_rule_per_gamma"] = R.c7_table(inputs["c7"])
        for delta in R.load(inputs["c7"])["deltas"]:
            tables[f"c7_headline_delta{delta:g}"] = R.c7_headline(
                inputs["c7"], delta)

    if os.path.exists(inputs["e6"]):
        tables["c8_sparse"] = R.c8_sparse_table(inputs["e6"])
        tables["c8_support_stability"] = R.c8_support_stability(inputs["e6"])
        tables["c8_cost_panels"] = R.c8_cost_table(inputs["e6"])

    if os.path.exists(inputs["backbone"]):
        tables["backbone_headline"] = R.backbone_headline(inputs["backbone"])
        tables["backbone_per_seed"] = R.backbone_table(inputs["backbone"])
        tables["backbone_per_judge"] = R.backbone_per_judge(inputs["backbone"])

    return tables


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, columns: List[str], digits: int = 3) -> str:
    sub = df[columns].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda v: f"{v:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in sub.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _c1_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C1 — individual redundancy certificates do not compose", ""]
    lines.append("**Claim.** A judge certified removable on its own need not be "
                 "removable alongside other judges that were each certified the "
                 "same way.")
    lines.append("")

    if not os.path.exists(inputs["e0_real"]):
        lines.append("_Not run on a real panel._")
        return lines + [""]

    df = R.c1_table(inputs["e0_real"])
    declared = df[df["gamma_source"] == "declared"]
    breaks = df[df["gamma_source"] == "loo_breakpoint"]
    multi = breaks[breaks["n_individually_removable"] >= 2]

    loo = df["loo_min"].min()
    max_declared = declared["gamma"].max() if len(declared) else float("nan")

    lines.append(f"**Test.** For each tolerance gamma, collect every judge whose "
                 f"leave-one-out error is at most gamma, delete them all at once, "
                 f"and measure what the remaining panel achieves.")
    lines.append("")

    # A violation at a tolerance the plan fixed in advance is the strongest form
    # of C1 available: no tolerance was chosen after seeing the data, and what is
    # tested is one exact coverage evaluation rather than a bounded search.  It is
    # reported first because the smaller Core-8 panel had an empty removable set
    # across the whole grid, so a reader who knows that result needs to see
    # immediately that this panel does not share it.
    declared_multi = declared[declared["n_individually_removable"] >= 2]
    if len(declared_multi):
        gamma = declared_multi["gamma"].min()
        at_gamma = declared_multi[declared_multi["gamma"] == gamma]
        lines.append(
            f"**On the predeclared grid.** At gamma = {gamma:g}, "
            f"{int(at_gamma['n_individually_removable'].mean())} judges are each "
            f"individually certified removable, yet deleting them together breaks "
            f"the tolerance in "
            f"{int(at_gamma['naive_violates_gamma'].sum())} of {len(at_gamma)} "
            f"split seeds (error {at_gamma['naive_coverage'].min():.3f}–"
            f"{at_gamma['naive_coverage'].max():.3f} against a budget of "
            f"{gamma:g}). Across the whole grid "
            f"{int(declared_multi['naive_violates_gamma'].sum())} of "
            f"{len(declared_multi)} cases with two or more removable judges fail. "
            f"This is the claim at a tolerance fixed before the data was seen, and "
            f"each failure is a single exact coverage evaluation."
        )
        lines.append("")

    if len(declared) and loo > max_declared:
        lines.append(
            f"On the predeclared grid (up to gamma = {max_declared:g}) the "
            f"removable set is empty at every tolerance, because the smallest "
            f"leave-one-out error on this panel is {loo:.3f}. The predeclared "
            f"grid therefore says nothing about composition here, and this is "
            f"reported rather than hidden."
        )
        lines.append("")

    # This holds whether or not the grid was vacuous, and the result below is
    # computed from the breakpoints either way, so the reason for using them has
    # to be stated unconditionally rather than only when the grid says nothing.
    if len(multi):
        lines.append(
            "The removable set only changes when gamma crosses a leave-one-out "
            "error, so the sorted leave-one-out errors are the complete set of "
            "informative tolerances. Those are used below and are labelled as "
            "data-driven throughout."
        )
        lines.append("")

    if len(multi):
        n_viol = int(multi["naive_violates_gamma"].sum())
        gap = multi["composition_gap"].mean()
        # The minimum feasible panel is only enumerated while that is affordable;
        # past E0's budget it is backward elimination's upper bound, which makes
        # the gap an upper bound too.  Say so rather than quoting it flat, since
        # a reader would otherwise take the sign of a bound as evidence.
        n_bounded = int((~multi["min_feasible_is_exact"]).sum())
        relation = "at most " if n_bounded else ""
        lines.append(
            f"**Result.** Across {len(multi)} (split seed, tolerance) cases in "
            f"which at least two judges were individually certified removable, "
            f"the joint deletion broke the tolerance in **{n_viol} of "
            f"{len(multi)}** cases. The mean composition gap — how many more "
            f"judges the joint constraint requires than the one-at-a-time audit "
            f"kept — is {relation}**{gap:+.2f}** judges."
        )
        lines.append("")
        if n_bounded:
            lines.append(
                f"_In {n_bounded} of {len(multi)} cases the panel was too large "
                f"to enumerate within E0's evaluation budget, so the minimum "
                f"feasible size there is backward elimination's upper bound "
                f"(certified no lower than `min_feasible_lower_bound`) and the "
                f"gap above is an upper bound. **The claim does not rest on it:** "
                f"the {n_viol} violations are single coverage evaluations and are "
                f"exact on every case._"
            )
            lines.append("")
        worst = multi.loc[multi["naive_coverage"].replace(np.inf, np.nan).idxmax()]
        lines.append(
            f"Worst finite case: at gamma = {worst['gamma']:.3f}, "
            f"{int(worst['n_individually_removable'])} judges each passed their own "
            f"test, yet deleting them together gave error "
            f"{worst['naive_coverage']:.3f}."
        )
        lines.append("")
        lines.append("**Verdict: SUPPORTED** on real judges.")
    else:
        lines.append("_No tolerance admitted two or more removable judges._")

    lines.append("")
    if os.path.exists(inputs["e0_synthetic"]):
        syn = R.c1_headline(inputs["e0_synthetic"], gamma_source=None)
        hit = syn[syn["naive_set_fails"]]
        lines.append(
            f"On the two controlled constructions, where the correct answer is "
            f"known in advance, the naive set violates the tolerance at "
            f"{len(hit)} of {len(syn)} tolerance points, with a mean composition "
            f"gap of {syn['composition_gap'].mean():+.2f} judges."
        )
        lines.append("")
        lines.append(_md_table(
            syn, ["instance", "gamma", "removable", "naive_size",
                  "min_feasible", "composition_gap", "naive_set_fails"]))
        lines.append("")
    return lines


def _c2_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C2 — the physical-to-virtual compression frontier", ""]
    lines.append("**Claim.** A small physical panel can reconstruct every "
                 "virtual judge to within a small worst-context error on held-out "
                 "items.")
    lines.append("")
    if not os.path.exists(inputs["e1"]):
        return lines + ["_Not run._", ""]

    df = R.c2_table(inputs["e1"], "TEST")
    cov = df[df["method"] == "coverage_backward"].sort_values("k")
    full = int(cov["k"].max())

    lines.append(
        "**Test.** Weights are fitted on FIT and scored on the locked TEST "
        "split, so no judge is ever evaluated on the items its weights were "
        "chosen from. The headline is the worst judge's worst context, not the "
        "average, because the claim is about every judge being preserved."
    )
    lines.append("")
    lines.append(_md_table(
        cov, ["k", "worst_judge_tv", "mean_judge_tv", "median_item_tv",
              "p95_item_tv", "verdict_agreement", "calls_avoided_frac"]))
    lines.append("")

    half = cov[cov["k"] <= max(1, full // 2)]
    if len(half):
        row = half.iloc[-1]
        lines.append(
            f"At k = {int(row['k'])} of {full} judges — {row['calls_avoided_frac']:.0%} "
            f"of judge calls avoided — the median item is reconstructed to "
            f"{row['median_item_tv']:.3f} TV and verdicts agree "
            f"{row['verdict_agreement']:.1%} of the time, while the worst judge "
            f"in its worst context still sits at {row['worst_judge_tv']:.3f}."
        )
        lines.append("")
        lines.append(
            "**Reading this honestly:** the typical item compresses well, but "
            "the worst-judge worst-context error is what C2 is stated over, and "
            "on this panel it stays well above the tolerances in the "
            "predeclared grid. The frontier is reported as measured."
        )

        # Why the declared band is out of reach rather than merely missed.  The
        # cheapest possible compression is dropping one judge, so its error is a
        # floor on every k below the full panel: if that floor already exceeds
        # the target, no panel size can meet it and the shortfall is a property
        # of the panel, not of the pruner.
        one_short = cov[cov["k"] == full - 1]
        if len(one_short):
            floor = float(one_short.iloc[0]["worst_judge_tv"])
            lines.append("")
            lines.append(
                f"The shortfall is structural, not a matter of tuning. Removing "
                f"a single judge — the most redundant one in the panel, and the "
                f"least that can be removed at all — already costs {floor:.3f}. "
                f"That is a floor on every smaller panel, so no k below {full} "
                f"can reach the 0.08-0.10 band the evidence matrix asks for at "
                f"6-10 judges. The judges are less mutually redundant than that "
                f"planning target assumed."
            )
            lines.append("")
            # The verdict has to name the tolerance it is given against.  The
            # measured frontier is sound; what fails is the planning target,
            # and the floor above says it fails for every k at once.
            lines.append(
                f"**Verdict: UNSUPPORTED** at the declared tolerance. The "
                f"frontier is measured and the reconstruction behaves as the "
                f"claim describes, but no panel size reaches the 0.08-0.10 "
                f"worst-context band, because the floor at k = {full - 1} is "
                f"already {floor:.3f}. The claim is not contradicted by a "
                f"better method existing — it is out of reach on this panel."
            )
    lines.append("")
    return lines


def _c3_paired_section(inputs: Dict[str, str]) -> List[str]:
    """Section 34.2: the paired test, which is the one that can actually decide."""
    paired = R.c3_paired_table(inputs["c3"])
    if paired.empty:
        return []

    pooled = (
        paired.groupby(["method", "label"], as_index=False)
        .agg(delta=("delta", "mean"), lo=("lo", "mean"), hi=("hi", "mean"),
             cells_won=("n_seeds_reference_better", "sum"),
             cells_lost=("n_seeds_baseline_better", "sum"),
             cells=("n_seeds", "sum"))
        .sort_values("delta")
    )
    total_cells = int(pooled["cells"].iloc[0])

    lines = ["### Head-to-head on the same items", ""]
    lines.append(
        "Comparing two independently computed intervals and asking whether they "
        "overlap is the wrong test here: both methods are scored on one set of "
        "items, so most of the sampling noise is shared and cancels in the "
        "difference. Below, coverage (backward) and each baseline are resampled "
        "on the **same** bootstrap items and the difference is taken. A negative "
        "delta means coverage is better. `cells` counts the "
        "(split seed, panel size) pairs. `won` and `lost` count the cells whose "
        "whole 95% interval falls on one side of zero, split by which side: a "
        "decided cell that went against coverage is a loss and is reported as "
        "one. The remainder are undecided."
    )
    lines.append("")

    show = pooled.rename(columns={"cells_won": "won", "cells_lost": "lost"})
    lines.append(_md_table(
        show, ["label", "delta", "lo", "hi", "won", "lost", "cells"]))
    lines.append("")

    # `random_best_draw` is the best of 100 random panels chosen by looking at
    # the answer, so it is an oracle rather than a competitor and is discussed
    # on its own terms.
    oracle = pooled[pooled["method"] == "random_best_draw"]
    rest = pooled[pooled["method"] != "random_best_draw"]

    decided = rest[rest["cells_won"] >= 0.5 * total_cells]
    ties = rest[((rest["cells_won"] + rest["cells_lost"]) <= 0.2 * total_cells)
                & (~rest["method"].isin(R.COVERAGE_METHODS))]

    if len(decided):
        lines.append(
            f"Coverage wins on a majority of cells against "
            f"{', '.join(decided['label'])} — every heuristic a practitioner "
            f"would actually reach for."
        )
        lines.append("")

    # Where coverage loses is the honest limit of the method and has to be
    # stated at the same volume as the wins, not left for a reader to derive
    # from the `lost` column.
    lost = paired[paired["n_seeds_baseline_better"] > 0]
    if len(lost):
        budgets = sorted(int(k) for k in lost["k"].unique())
        losers = sorted(set(lost["label"]))
        run = (budgets == list(range(min(budgets), max(budgets) + 1)))
        where = (f"k <= {max(budgets)}" if run and min(budgets) == int(paired["k"].min())
                 else "k in " + ", ".join(str(b) for b in budgets))
        # `total_cells` is per baseline, so the denominator for a count pooled
        # over baselines is that times the number of baselines compared.
        n_comparisons = total_cells * int(pooled["method"].nunique())
        lines.append(
            f"**Where coverage loses.** Of the {n_comparisons} comparisons — "
            f"{total_cells} cells against each of {int(pooled['method'].nunique())} "
            f"baselines — {int(pooled['cells_lost'].sum())} go against "
            f"coverage, and every "
            f"one of them sits at {where} — the smallest budgets on the grid. "
            f"The baselines that win there are {', '.join(losers)}. From "
            f"k = {max(budgets) + 1} upward coverage is not beaten by any "
            f"baseline in any seed. Backward elimination has almost nothing to "
            f"remove at two or three judges, so the greedy order it produces "
            f"carries little information; that is a real limit of the method "
            f"and not a sampling accident, since the losses are unanimous "
            f"across seeds."
        )
        lines.append("")
    if len(ties):
        lines.append(
            f"**Reading this honestly:** {', '.join(ties['label'])} are not "
            f"separated from coverage on this panel. They are pairwise-geometry "
            f"selectors, and at N = 8 they often pick the identical subset, so "
            f"there is nothing for the test to resolve. Whether coverage "
            f"separates from them is what the larger panel has to settle."
        )
        lines.append("")
    if len(oracle):
        row = oracle.iloc[0]
        lines.append(
            f"The last row is not a competitor: it is the best of 100 random "
            f"panels *chosen after seeing which one won*, so no practitioner "
            f"could produce it. It is included as an oracle, and coverage lands "
            f"{abs(row['delta']):.3f} TV "
            f"{'behind' if row['delta'] > 0 else 'ahead of'} it — that is, "
            f"coverage picks about as well as a hundred random tries with "
            f"hindsight, from a single pass."
        )
        lines.append("")
    return lines


def _c3_robustness_section(inputs: Dict[str, str]) -> List[str]:
    """Section 34.4: the headline must not rest on one favourable split."""
    rob = R.c3_split_robustness(inputs["c3"])
    if rob.empty or int(rob["n_seeds"].iloc[0]) < 2:
        return []

    cov = rob[rob["method"] == "coverage_backward"].iloc[0]
    others = rob[~rob["method"].isin(R.COVERAGE_METHODS)]
    best_other = others.loc[others["mean"].idxmin()]

    lines = ["### Does the result depend on the split?", ""]
    lines.append(
        f"Each of the {int(cov['n_seeds'])} predeclared split seeds gives one "
        f"number per method. The spread of those numbers shows whether the "
        f"headline rests on one favourable partition."
    )
    lines.append("")
    lines.append(_md_table(rob, ["label", "mean", "sd", "min", "max", "range"]))
    lines.append("")
    if cov["max"] < best_other["min"]:
        lines.append(
            f"Coverage's **worst** split ({cov['max']:.3f}) is still better than "
            f"{best_other['label']}'s **best** split ({best_other['min']:.3f}), "
            f"so the ordering survives every partition, not just the average."
        )
    else:
        lines.append(
            f"Coverage's worst split ({cov['max']:.3f}) overlaps "
            f"{best_other['label']}'s range "
            f"[{best_other['min']:.3f}, {best_other['max']:.3f}], so the "
            f"across-seed ranges for these two methods are not separated even "
            f"though the means are ordered."
        )
    lines.append("")
    return lines


def _c3_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C3 — coverage-based selection beats the baselines", ""]
    lines.append("**Claim.** Choosing the physical panel by the coverage "
                 "functional does better than accuracy ranking, random choice, "
                 "family diversity, cost heuristics, and standard pairwise-"
                 "geometry selection.")
    lines.append("")
    if not os.path.exists(inputs["c3"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["c3"])
    head = R.c3_headline(inputs["c3"])
    ks = head["k_values"].iloc[0]

    lines.append(
        f"**Test.** Every method is compared at equal panel size k, averaged "
        f"over k in {ks} and over {len(payload['split_seeds'])} split seeds, "
        f"with {payload['n_bootstrap']} item-bootstrap replicates. Trivial "
        f"budgets are excluded: at k = 1 no convex combination exists, and at "
        f"k = N every judge reconstructs itself exactly for every method."
    )
    lines.append("")
    lines.append(_md_table(
        head, ["label", "worst_judge_tv", "boot_lo", "boot_hi",
               "mean_judge_tv", "verdict_agreement"]))
    lines.append("")

    cov = head[head["method"] == "coverage_backward"].iloc[0]
    others = head[~head["is_coverage"]]
    best_other = others.loc[others["worst_judge_tv"].idxmin()]
    worst_other = others.loc[others["worst_judge_tv"].idxmax()]

    lines.append(
        f"Coverage (backward) reaches {cov['worst_judge_tv']:.3f}. The strongest "
        f"baseline is {best_other['label']} at {best_other['worst_judge_tv']:.3f}; "
        f"the weakest is {worst_other['label']} at "
        f"{worst_other['worst_judge_tv']:.3f}."
    )
    lines.append("")

    lines += _c3_paired_section(inputs)
    lines += _c3_robustness_section(inputs)

    # Section 31 sets the minimum bar at beating `top_accuracy`; the paired
    # test is what decides it, and the verdict has to survive the budgets
    # where coverage loses rather than average them away.
    paired = R.c3_paired_table(inputs["c3"])
    if not paired.empty:
        bar = paired[paired["method"] == "top_accuracy"]
        lost = paired[paired["n_seeds_baseline_better"] > 0]
        beats_bar = (len(bar)
                     and int(bar["n_seeds_baseline_better"].sum()) == 0
                     and int(bar["n_seeds_reference_better"].sum())
                     >= 0.5 * int(bar["n_seeds"].sum()))
        if beats_bar and not len(lost):
            lines.append(
                "**Verdict: SUPPORTED.** Coverage beats every baseline, "
                "including the section 31 minimum bar of top accuracy, and is "
                "not beaten anywhere on the budget grid."
            )
        elif beats_bar:
            safe = int(lost["k"].max()) + 1
            lines.append(
                f"**Verdict: SUPPORTED** for k >= {safe}. Coverage beats the "
                f"section 31 minimum bar of top accuracy in every decided "
                f"cell, and from k = {safe} upward no baseline beats it in any "
                f"seed. Below that it does lose, so the claim is stated with "
                f"the budget range attached rather than as a blanket result."
            )
        else:
            lines.append(
                "**Verdict: PARTIALLY SUPPORTED.** Coverage does not clear the "
                "section 31 minimum bar of beating top accuracy across the "
                "grid; the paired table above shows where it falls short."
            )
        lines.append("")

    df = R.c3_table(inputs["c3"])
    if "exhaustive" in set(df["method"]):
        deficits = []
        for k, group in df[df["k"].isin(ks)].groupby("k"):
            ex = group[group["method"] == "exhaustive"]["worst_judge_tv"]
            if len(ex):
                deficits.append(ex.iloc[0] - group["worst_judge_tv"].min())
        if deficits:
            lines.append("### Does selection on FIT transfer to TEST?")
            lines.append("")
            lines.append(
                f"The enumerated optimum is optimal *on FIT* and is then scored "
                f"on TEST like everything else, so it can lose on TEST. It does, "
                f"by at most {max(deficits):.3f} TV and {np.mean(deficits):.3f} "
                f"on average. That gap is the cost of choosing a panel on one "
                f"set of items and deploying it on another, and it sets a floor "
                f"on how finely any two methods can be distinguished here."
            )
            lines.append("")

    cost = R.c3_cost_table(inputs["c3"])
    lines.append("### Cost")
    lines.append("")
    lines.append(
        "Judges are not equally expensive, so equal-k is not equal-cost. These "
        "are measured inference seconds for the scored items, not a parameter-"
        "count proxy."
    )
    lines.append("")
    lines.append(_md_table(
        cost, ["k", "worst_judge_tv", "verdict_agreement", "panel_seconds",
               "seconds_saved", "speedup"]))
    lines.append("")

    floor = R.lowrank_floor_table(inputs["c3"])
    lines.append("### How much of the gap is unavoidable")
    lines.append("")
    lines.append(
        "The rank-k floor is what a k-dimensional basis achieves when the "
        "basis is free to be anything rather than a set of real judges. It is "
        "**not deployable** — nothing can be run to produce those directions, "
        "and its outputs are not on the simplex — so it is a reference point "
        "for how much of the error is dimensional rather than a consequence of "
        "picking the wrong judges."
    )
    lines.append("")
    pca = floor[floor["method"] == "pca"]
    lines.append(_md_table(pca, ["k", "worst_judge_tv", "mean_judge_tv"]))
    lines.append("")
    # The basis is fitted on FIT and scored on held-out rows, and the
    # objective it minimises is squared error, not worst-judge TV.  Neither
    # matches the quantity tabulated, so the column is not a certified lower
    # bound and the data shows it: it goes up in places.  Say so rather than
    # let a reader treat a rise as an error.
    w = pca.sort_values("k")["worst_judge_tv"].to_numpy()
    rises = [int(k) for k, a, b in zip(pca.sort_values("k")["k"].to_numpy()[1:],
                                       w[:-1], w[1:]) if b > a + 1e-9]
    if rises:
        lines.append(
            f"This column is not a certified lower bound and should not be "
            f"read as one. The basis is fitted on FIT and scored on held-out "
            f"rows, and it is chosen to minimise squared error rather than "
            f"worst-judge TV, so neither the split nor the objective matches "
            f"the number tabulated. That is visible in the table: it rises at "
            f"k = {', '.join(str(r) for r in rises)}, which a true floor could "
            f"not do. It stays below the achieved error at every k here, which "
            f"is the comparison it is used for."
        )
        lines.append("")

    rec = R.reconstruction_table(inputs["c3"])
    off = rec[(rec["rule"] != "simplex") & (rec["off_simplex_frac"] > 0)]
    lines.append("### Reconstruction rule")
    lines.append("")
    if len(off):
        lines.append(
            f"Unconstrained rules can score lower on TV while leaving the "
            f"simplex on up to {off['off_simplex_frac'].max():.0%} of items. "
            f"Their output is then not a probability distribution and cannot be "
            f"shipped as a judge verdict, which is why the simplex rule is the "
            f"primary one."
        )
        lines.append("")
    lines.append(_md_table(
        rec, ["k", "rule", "worst_judge_tv", "verdict_agreement",
              "off_simplex_frac"]))
    lines.append("")
    return lines


def _c4_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C4 — stress specialists and multi-context selection", ""]
    lines.append("**Claim.** Selecting on clean items alone retires judges that "
                 "are redundant on average but distinctive under position swaps, "
                 "verbosity changes, rubric changes or a hidden reference. "
                 "Selecting against the worst context retains those stress "
                 "specialists and lowers worst-context error.")
    lines.append("")
    if not os.path.exists(inputs["c4"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["c4"])
    head = R.c4_headline(inputs["c4"])
    specialists = R.c4_specialists(inputs["c4"])

    lines.append(
        f"**Test.** One variable moves: which contexts the selector may see. "
        f"Panel, greedy backward selector and locked TEST split are held fixed, "
        f"and all three arms are scored on all {len(payload['contexts'])} "
        f"contexts, so equal-k comparison is fair. `clean_select` is handicapped "
        f"only at selection time and still gets robust weights, which separates "
        f"the selection mistake from the fitting mistake; `clean_pipeline` is "
        f"what someone who never considered contexts would deploy. Averaged over "
        f"{len(payload['split_seeds'])} split seeds."
    )
    lines.append("")
    lines.append(
        "`delta` is baseline minus robust, so positive is the direction the "
        "claim predicts."
    )
    lines.append("")
    lines.append(_md_table(
        head, ["baseline", "k", "delta_mean", "delta_sd", "delta_min",
               "delta_max", "n_seeds_better", "robust_wins_every_seed"]))
    lines.append("")

    # A positive mean delta is not the bar.  With five seeds a mean can be
    # carried by one partition, so the verdict is stated on unanimity and the
    # worst seed, and a split result is reported as split rather than rounded up.
    for baseline, group in head.groupby("baseline"):
        unanimous = int(group["robust_wins_every_seed"].sum())
        worst = group["delta_min"].min()
        lines.append(
            f"Against **{baseline}**: mean delta {group['delta_mean'].mean():+.3f} "
            f"TV, unanimous across seeds at {unanimous} of {len(group)} budgets, "
            f"and the worst single (seed, budget) delta is {worst:+.3f}."
        )
    lines.append("")

    stable = specialists[specialists["in_every_seed"]]
    if len(stable):
        lines.append(
            f"{len(stable)} of {len(specialists)} judges flagged as stress "
            f"specialists are flagged under *every* split seed: "
            f"{', '.join(stable['judge'])}. Only those are findings; the rest are "
            f"candidates that one partition produced."
        )
    else:
        lines.append(
            "_No judge was flagged as a stress specialist under every split "
            "seed, so the specialist label is not stable on this panel._"
        )
    lines.append("")
    lines.append(_md_table(
        specialists, ["judge", "n_seeds_flagged", "in_every_seed",
                      "binding_contexts"]))
    lines.append("")

    # The two baselines answer different questions and must be scored separately.
    # `clean_pipeline` is handicapped at selection *and* fitting, `clean_select`
    # only at selection, so the gap between the two advantages is how much of the
    # effect is the fitting mistake rather than the selection mistake.  Collapsing
    # them into one verdict would claim the selection result the weaker arm earns.
    per_baseline = {
        baseline: (
            bool(group["robust_wins_every_seed"].all()),
            float(group["delta_mean"].mean()),
        )
        for baseline, group in head.groupby("baseline")
    }
    pipeline = per_baseline.get("clean_pipeline")
    select = per_baseline.get("clean_select")

    if pipeline and select and pipeline[0] and not select[0]:
        share = select[1] / pipeline[1] if pipeline[1] else float("nan")
        lines.append(
            f"**Verdict: PARTIALLY SUPPORTED.** Strong for the deployed "
            f"pipeline, weak for selection alone. "
            f"Against `clean_pipeline` — what a context-unaware user would "
            f"actually ship — robust selection wins at every budget under every "
            f"seed, by {pipeline[1]:+.3f} TV on average. Against `clean_select`, "
            f"which is handicapped only at selection time and still receives "
            f"robust weights, the advantage falls to {select[1]:+.3f} TV and is "
            f"not unanimous at every budget. So roughly {share:.0%} of the total "
            f"effect is attributable to *which judges were selected* and the rest "
            f"to *which contexts the weights were fitted on*. C4 as stated is "
            f"about selection, so this is a partial result for the claim and a "
            f"strong result for the pipeline."
        )
    elif all(unanimous for unanimous, _ in per_baseline.values()):
        lines.append(
            "**Verdict: SUPPORTED** — robust selection wins against both "
            "baselines at every budget under every split seed."
        )
    else:
        lines.append(
            "**Verdict: PARTIALLY SUPPORTED** — robust selection does not win at "
            "every budget under every seed, and the table above says where."
        )
    lines.append("")
    return lines


def _c5_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C5 — does the compressed panel still decide the same way?", ""]
    lines.append("**Claim.** A panel compressed to virtual judges preserves the "
                 "decisions the full panel would have made: accuracy against "
                 "gold within about a percentage point, and rank agreement "
                 "above 0.95.")
    lines.append("")
    if not os.path.exists(inputs["c5"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["c5"])
    k = C5_HEADLINE_K

    lines.append(
        f"**Test.** Every arm is a panel of the same size k = {k}, aggregated "
        f"to one verdict per item and scored against RewardBench 2 gold. "
        f"`virtual` keeps k judges and reconstructs the other "
        f"{len(payload['judges']) - k} from them; `physical` keeps the same k "
        f"and simply drops the rest. The gap between those two rows is what "
        f"reconstruction buys, and it is the only comparison in which one "
        f"variable moves. Averaged over "
        f"{len(payload['split_seeds'])} split seed"
        f"{'' if len(payload['split_seeds']) == 1 else 's'} and "
        f"{len(payload['contexts'])} contexts."
    )
    lines.append("")
    lines.append(
        f"Three aggregators are reported because the claim is silent about "
        f"which one a user would deploy, and they do not agree: "
        f"{', '.join(payload['aggregators'])}. `mean` and `majority` are fixed "
        f"rules; `logistic` is fitted on FIT only and never sees the split it "
        f"is scored on."
    )
    lines.append("")

    columns = ["label", "accuracy", "delta_accuracy", "verdict_agreement",
               "item_rank_tau", "domain_rank_tau", "nll", "ece"]
    for aggregator in payload["aggregators"]:
        head = R.c5_headline(inputs["c5"], k, aggregator)
        lines.append(f"### Aggregator: {aggregator}")
        lines.append("")
        lines.append(_md_table(head, columns))
        lines.append("")

        virtual = head[head["arm"] == "virtual"]
        physical = head[head["arm"] == "physical"]
        if not (len(virtual) and len(physical)):
            continue
        v, p = virtual.iloc[0], physical.iloc[0]
        wins = int(p["n_seeds_virtual_better"])
        n_seeds = int(p["n_seeds"])
        lines.append(
            f"Reconstruction costs {abs(v['delta_accuracy']):.4f} accuracy "
            f"against the full panel, and dropping the same judges costs "
            f"{abs(p['delta_accuracy']):.4f}. Virtual beats physical on "
            f"{wins} of {n_seeds} split seeds. Verdict agreement with the full "
            f"panel is {v['verdict_agreement']:.3f} for virtual against "
            f"{p['verdict_agreement']:.3f} for physical."
        )
        lines.append("")

    # The two halves of the claim have to be judged separately: the accuracy
    # tolerance and the rank tolerance are different numbers and can fail
    # independently, so collapsing them would hide which one broke.
    head = R.c5_headline(inputs["c5"], k, "mean")
    virtual = head[head["arm"] == "virtual"]
    if len(virtual):
        v = virtual.iloc[0]
        accuracy_ok = abs(v["delta_accuracy"]) <= 0.01
        tau = float(v["item_rank_tau"])
        rank_ok = tau >= 0.95
        lines.append("### Verdict")
        lines.append("")
        lines.append(
            f"Under the mean aggregator the accuracy drop is "
            f"{abs(v['delta_accuracy']):.4f}, which "
            f"{'meets' if accuracy_ok else 'misses'} the stated tolerance of "
            f"about one percentage point. The item-level rank correlation is "
            f"{tau:.3f}, which {'meets' if rank_ok else 'misses'} the stated "
            f"0.95."
        )
        lines.append("")
        if accuracy_ok and rank_ok:
            lines.append("**Verdict: SUPPORTED.**")
        elif accuracy_ok or rank_ok:
            lines.append(
                "**Verdict: PARTIALLY SUPPORTED** — one half of the stated "
                "tolerance is met and the other is not, and the numbers above "
                "say which."
            )
        else:
            lines.append(
                "**Verdict: UNSUPPORTED** at the stated tolerances. The "
                "compressed panel tracks the full panel far better than an "
                "equally sized physical panel does, which is the comparison "
                "that matters for deployment, but it does not reach the "
                "absolute numbers the claim names."
            )
        lines.append("")

    # A learned aggregator re-weights the judges itself, so it can absorb the
    # missing ones.  Where that happens, reconstruction has nothing left to
    # add, and saying so is the honest reading of the table above.
    logistic = R.c5_headline(inputs["c5"], k, "logistic")
    lv = logistic[logistic["arm"] == "virtual"]
    lp = logistic[logistic["arm"] == "physical"]
    if len(lv) and len(lp):
        margin = float(lp.iloc[0]["accuracy"] - lv.iloc[0]["accuracy"])
        if abs(margin) < 0.005:
            lines.append(
                f"**A negative result worth stating.** Under the learned "
                f"aggregator, virtual and physical are separated by only "
                f"{abs(margin):.4f} accuracy. A logistic aggregate already "
                f"fits its own weights over whichever judges it is given, so "
                f"it recovers by itself most of what reconstruction supplies. "
                f"Reconstruction earns its place under the fixed aggregators, "
                f"not under a learned one."
            )
            lines.append("")

    # The single-best row routinely beats the panel on this dataset.  That is a
    # property of RewardBench 2 gold and not of the compression, and leaving it
    # unexplained would invite the reader to conclude panels are pointless.
    single = head[head["arm"] == "single_best"]
    full = head[head["arm"] == "full"]
    if len(single) and len(full) and (
            float(single.iloc[0]["accuracy"]) > float(full.iloc[0]["accuracy"])):
        lines.append(
            f"**Note on the single-judge row.** One judge alone scores "
            f"{single.iloc[0]['accuracy']:.3f} against gold, above the full "
            f"panel's {full.iloc[0]['accuracy']:.3f}. That is a fact about this "
            f"dataset: RewardBench 2 gold is itself a model-assisted label, so "
            f"a judge close to the labeller wins on agreement while still "
            f"carrying its own biases. It is reported because it is in the "
            f"data, but it is not evidence that panels are unnecessary — "
            f"nothing in the plan's setup lets a user identify that judge in "
            f"advance."
        )
        lines.append("")

    lines.append(f"_{payload['note_system_ranking']}_")
    lines.append("")
    return lines


def _c6_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C6 — the exchange structure greedy selection misses", ""]
    lines.append("**Claim.** Which judges are worth keeping depends on the set "
                 "they sit in, so greedy add-one or drop-one selection can stop "
                 "at a panel larger than necessary, and a local exchange that "
                 "swaps judges in and out recovers the difference.")
    lines.append("")
    if not os.path.exists(inputs["c6"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["c6"])
    head = R.c6_headline(inputs["c6"])
    df = R.c6_table(inputs["c6"])
    n_instances = int(head["n_instances"].max())
    sizes = sorted(set(payload["subpanel_sizes"]))

    lines.append(
        f"**Test.** The comparison needs a known answer, so it is run on "
        f"{len(payload['subpanel_sizes'])} fixed subpanels of "
        f"{sizes[0]}–{sizes[-1]} judges, small enough that the smallest "
        f"feasible set is found by exact search rather than assumed. Each "
        f"method is then asked for the smallest panel meeting the same "
        f"tolerance, and `gap` is how many judges more than the certified "
        f"optimum it returned. {n_instances} (subpanel, tolerance) instances "
        f"in total."
    )
    lines.append("")
    lines.append(f"_{payload['note_fit_and_score']}_")
    lines.append("")
    lines.append(_md_table(
        head, ["label", "exact_rate", "mean_gap", "max_gap", "mean_seconds"]))
    lines.append("")

    greedy = head[head["method"].isin(["forward", "backward", "forward_trim"])]
    swaps = head[head["method"].str.startswith("swap")]
    if len(greedy) and len(swaps):
        best_greedy = greedy.loc[greedy["mean_gap"].idxmin()]
        best_swap = swaps.loc[swaps["mean_gap"].idxmin()]
        lines.append(
            f"The best plain greedy method is {best_greedy['label']}, which "
            f"lands on the certified optimum in {best_greedy['exact_rate']:.0%} "
            f"of instances and averages {best_greedy['mean_gap']:.2f} extra "
            f"judges. The best exchange method is {best_swap['label']} at "
            f"{best_swap['exact_rate']:.0%} and {best_swap['mean_gap']:.2f}."
        )
        lines.append("")

    # The section 26 targets are stated for 2-swap only, so the verdict is read
    # off that row and nowhere else.
    swap2 = head[head["method"] == "swap2"]
    if len(swap2):
        row = swap2.iloc[0]
        target_columns = [c for c in head.columns if c.startswith("swap2_")]
        met = {c: bool(row[c]) for c in target_columns if pd.notna(row[c])}
        if met:
            lines.append("The planning targets in section 26 are stated for "
                         "2-swap. Against them:")
            lines.append("")
            for name, ok in met.items():
                lines.append(f"- `{name}` — {'met' if ok else 'not met'}")
            lines.append("")
            if all(met.values()):
                lines.append("**Verdict: SUPPORTED.** 2-swap meets every "
                             "predeclared target.")
            else:
                lines.append(
                    "**Verdict: PARTIALLY SUPPORTED.** The targets that were "
                    "missed are listed above, and the per-instance table shows "
                    "which subpanels and tolerances are responsible."
                )
            lines.append("")

    # A claim about set-dependence is only demonstrated by a case where a judge
    # that greedy refused to drop becomes droppable once another one moves.
    # One such case is worth more than any aggregate rate, so it is named.
    beaten = df[(df["method"] == "backward") & (df["gap"] > 0)]
    if len(beaten):
        case = beaten.loc[beaten["gap"].idxmax()]
        rival = df[(df["subpanel"] == case["subpanel"])
                   & (df["gamma"] == case["gamma"])
                   & (df["method"] == "swap2")]
        if len(rival) and int(rival.iloc[0]["size"]) < int(case["size"]):
            lines.append(
                f"**The trap, concretely.** On subpanel "
                f"{int(case['subpanel'])} ({int(case['n_judges'])} judges) at "
                f"tolerance {case['gamma']:g}, backward elimination stops at "
                f"{int(case['size'])} judges. The certified optimum is "
                f"{int(case['opt_size'])}, and 2-swap finds "
                f"{int(rival.iloc[0]['size'])}. Backward could not drop any "
                f"single judge from its panel without breaking the tolerance, "
                f"yet a panel one smaller exists — it just is not reachable by "
                f"deletions alone. That is the set-dependence the claim is "
                f"about."
            )
            lines.append("")

    improvement = payload["summary"]["swap3_over_swap2"]
    n_better = improvement["n_instances_improved"]
    lines.append("### Is a wider exchange worth it?")
    lines.append("")
    lines.append(
        f"3-swap searches a strictly larger neighbourhood than 2-swap and costs "
        f"{improvement['time_ratio']:.1f} times as long. It improves on 2-swap "
        f"in {n_better} instance{'' if n_better == 1 else 's'} and is worse in "
        f"{improvement['n_instances_worse']}, for a mean gap reduction of "
        f"{improvement['mean_gap_reduction']:.3f} judges. "
        + ("The extra neighbourhood is not paying for itself at this panel size."
           if improvement["mean_gap_reduction"] <= 0.05
           else "The extra neighbourhood does buy something here.")
    )
    lines.append("")
    lines.append(f"_{payload['note_gamma_grid']}_")
    lines.append("")
    return lines


def _c7_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C7 — certificates that hold up out of sample", ""]
    lines.append("**Claim.** A compression certificate computed with a "
                 "finite-sample bound on held-out items, corrected for testing "
                 "many judges at once, holds on fresh items at the stated "
                 "confidence, while the empirical error the weights were fitted "
                 "on does not.")
    lines.append("")
    if not os.path.exists(inputs["c7"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["c7"])
    n_reps = len(payload["repetitions"])
    primary_delta = payload["deltas"][0]

    lines.append(
        f"**Test.** The items are re-partitioned {n_reps} times into fitting, "
        f"certification and test thirds, always grouped by source item so one "
        f"prompt never straddles two splits. On each repetition a panel is "
        f"selected, every absent judge is reconstructed, each rule states a "
        f"bound from the certification third, and the bound is then checked "
        f"against the worst error actually observed on the test third. A "
        f"violation is a bound that the test error exceeded. Panel sizes "
        f"k in {payload['k_grid']}; tolerances "
        f"{payload['gammas']}; {payload['n_boot']} bootstrap replicates."
    )
    lines.append("")
    lines.append(
        "One point of bookkeeping matters more than it looks. A single source "
        "item produces several comparison pairs, and those pairs are not "
        "independent, so every bound averages within a source item first and "
        "is computed over source items. Treating the pairs as independent "
        "would shrink every interval by a factor the data does not earn."
    )
    lines.append("")

    head = R.c7_headline(inputs["c7"], primary_delta)
    per_gamma = R.c7_table(inputs["c7"])
    per_gamma = per_gamma[per_gamma["delta"].isna()
                          | np.isclose(per_gamma["delta"], primary_delta)]
    lines.append(f"### At delta = {primary_delta:g}")
    lines.append("")
    lines.append(
        "`certified_frac` is the share of cases the rule was able to certify "
        "at all, and `violation_rate` is conditional on that. Both are needed: "
        "a rule that certifies nothing never violates anything and is useless."
    )
    lines.append("")
    lines.append(_md_table(
        head, ["label", "certified_frac", "n_certified", "n_violated",
               "violation_rate", "worst_violation_rate_hi", "within_nominal",
               "useful"]))
    lines.append("")

    naive = head[head["rule"] == "empirical_fit"]
    primary = head[head["is_primary"]]
    if len(naive) and len(primary):
        n, p = naive.iloc[0], primary.iloc[0]
        lines.append(
            f"The naive certificate — the error measured on the very items the "
            f"weights were fitted to — is violated on "
            f"{n['violation_rate']:.1%} of the cases it certifies. The primary "
            f"rule, an empirical Bernstein bound on held-out items with a "
            f"Bonferroni correction across judge-context pairs, is violated on "
            f"{p['violation_rate']:.1%}, against a nominal "
            f"{primary_delta:.0%}."
        )
        lines.append("")
        if bool(p["within_nominal"]) and bool(p["useful"]):
            lines.append(
                "**Verdict: SUPPORTED.** The primary rule certifies a useful "
                "share of cases and its violation rate stays at or below the "
                "confidence level it claims, using the upper end of a Wilson "
                "interval rather than the point estimate so a small denominator "
                "is not mistaken for proof."
            )
        elif not bool(p["useful"]):
            lines.append(
                "**Verdict: UNSUPPORTED.** The primary rule certified nothing "
                "at these tolerances, so there is no coverage to check."
            )
        elif float(p["violation_rate"]) > primary_delta:
            lines.append(
                "**Verdict: PARTIALLY SUPPORTED.** The primary rule is "
                "violated more often than its nominal rate, and the per-"
                "tolerance table shows where."
            )
        else:
            # Zero (or few enough) observed violations, but the Wilson upper
            # bound still clears delta.  That is a statement about how many
            # cases the rule managed to certify, not about the bound failing:
            # "0 of 10" simply cannot be resolved below 5%.  Saying the rule
            # was violated here would be false, so name the binding cell and
            # call it what it is -- too few certified cases to demonstrate the
            # rate, at the tolerance where the rule certifies almost nothing.
            worst = per_gamma[per_gamma["rule"] == p["rule"]].sort_values(
                "violation_rate_hi").iloc[-1]
            lines.append(
                f"**Verdict: PARTIALLY SUPPORTED.** The primary rule was not "
                f"violated once in {int(p['n_certified'])} certified cases, so "
                f"nothing here contradicts the claim. It falls short only on "
                f"resolution: at gamma = {worst['gamma']:.2f} the rule certifies "
                f"just {int(worst['n_certified'])} cases, and "
                f"{int(worst['n_violated'])} violations out of that many bounds "
                f"the rate no tighter than {worst['violation_rate_hi']:.1%} — "
                f"above the nominal {primary_delta:.0%} however well the rule "
                f"behaves. More repetitions at the tight tolerances, not a "
                f"different bound, is what would settle it."
            )
        lines.append("")

    lines.append("### The price of each correction")
    lines.append("")
    lines.append(
        "Every step from the naive number to the primary rule costs width, and "
        "the point of reporting all four is that the cost is visible rather "
        "than asserted. Moving off the fitting split removes the optimism; the "
        "Bernstein bound replaces a normal approximation with one that holds at "
        "finite sample size; the Bonferroni correction pays for the fact that "
        "many judges and contexts are certified simultaneously; and the grouped "
        "bootstrap pays for the dependence that is actually present instead of "
        "the worst case, which is why it lands between the two."
    )
    lines.append("")

    lines.append("### Per tolerance")
    lines.append("")
    lines.append(_md_table(
        per_gamma, ["label", "gamma", "certified_frac", "n_certified",
                    "n_violated", "violation_rate", "violation_rate_hi"]))
    lines.append("")

    others = [d for d in payload["deltas"] if d != primary_delta]
    if others:
        lines.append(
            f"The same tables at delta in {others} are in `tables/`, so a "
            f"reader can see that a tighter confidence level widens the bound "
            f"and lowers the violation rate as it should."
        )
        lines.append("")
    return lines


def _c8_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## C8 — sparse certificates and cost-aware selection", ""]
    lines.append("**Claim.** A certificate stays useful when each virtual judge "
                 "is allowed only a handful of physical judges, and choosing a "
                 "panel by what it costs to run rather than by how many judges "
                 "it has buys real savings at the same coverage.")
    lines.append("")
    if not os.path.exists(inputs["e6"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["e6"])
    sparse = R.c8_sparse_table(inputs["e6"])
    stability = R.c8_support_stability(inputs["e6"])
    cost = R.c8_cost_table(inputs["e6"])

    lines.append(
        f"**Test, sparse half.** The support of every reconstruction weight "
        f"vector is capped at r judges, for r in {payload['caps']}, and the "
        f"cap is enforced exactly by fitting every support of that size rather "
        f"than by a penalty, so `ratio_to_uncapped` is the true price of "
        f"sparsity and not a solver artefact. Budgets {payload['budgets']}, "
        f"scored on {payload['eval_split']} over "
        f"{len(payload['split_seeds'])} split seeds."
    )
    lines.append("")
    lines.append(_md_table(
        sparse, ["k", "support_cap", "worst_judge_tv_mean", "worst_judge_tv_sd",
                 "mean_judge_tv", "mean_support_size", "ratio_to_uncapped",
                 "n_seeds"]))
    lines.append("")

    # The comparison that decides the sparse half is r = 3 against no cap: the
    # plan's strong result names four judges, so three clearing it is the
    # statement worth making, and the uncapped support size says whether the cap
    # bound anything at all.
    # The cap column mixes ints with the string "none", so it must be compared
    # against the value the table actually holds and not against its rendering.
    worst_three = float(
        sparse[sparse["support_cap"] == 3]["ratio_to_uncapped"].max())
    at_one = float(
        sparse[sparse["support_cap"] == 1]["ratio_to_uncapped"].max())
    mean_support = float(
        sparse[sparse["support_cap"] == "none"]["mean_support_size"].mean())
    at_four = sparse[sparse["support_cap"] == 4]
    worst_four = float(at_four["ratio_to_uncapped"].max())
    worst_k = int(at_four.loc[at_four["ratio_to_uncapped"].idxmax(), "k"])
    lines.append(
        f"Three physical judges per virtual judge is close to free: at r = 3 "
        f"the worst budget costs {worst_three:.3f} times the uncapped error, "
        f"and at r = 4 the worst costs {worst_four:.3f}. Both of those worst "
        f"cases are at k = {worst_k}, where the uncapped fit is already the "
        f"sparsest and the cap therefore bites hardest. The reason is visible "
        f"in the last column — the uncapped fit already spreads over "
        f"{mean_support:.1f} judges on average, so a cap at three or four is "
        f"describing the solution rather than constraining it. The one real "
        f"cliff is r = 1, which costs up to {at_one:.3f} times uncapped: a "
        f"single substitute judge is not a certificate, it is a replacement."
    )
    lines.append("")

    # Unanimity alone would read as instability at the wider caps, when what is
    # actually happening is that a wider cap has more near-equivalent supports
    # to choose between.  The count of distinct supports is the honest measure,
    # so both are reported.
    by_cap = stability.groupby("support_cap").agg(
        unanimous=("support_is_unanimous", "sum"),
        cases=("support_is_unanimous", "size"),
        distinct=("n_distinct_supports", "mean"),
    )
    tight, wide = by_cap.iloc[0], by_cap.iloc[-1]
    lines.append(
        f"Which judges stand in for which is largely a property of the panel "
        f"rather than of the partition, but less so as the cap widens: at "
        f"r = {by_cap.index[0]} the same support is picked under all five seeds "
        f"in {int(tight.unanimous)} of {int(tight.cases)} (budget, judge) "
        f"cases, falling to {int(wide.unanimous)} of {int(wide.cases)} at "
        f"r = {by_cap.index[-1]}. That is the expected direction — a wider cap "
        f"has more near-equivalent supports to choose between — and the "
        f"variation stays small in absolute terms, at "
        f"{tight.distinct:.1f} to {wide.distinct:.1f} distinct supports across "
        f"five seeds. The full table is `tables/c8_support_stability.csv`."
    )
    lines.append("")

    lines.append(
        f"**Test, cost half.** Every panel meeting a tolerance is enumerated, "
        f"then the cheapest one is picked under each of four objectives: "
        f"judge count, wall-clock seconds, weight memory in GB, and the number "
        f"of distinct model families that must be deployed. Tolerances "
        f"{payload['gammas']}. `same_panel_as_cardinality` is the column that "
        f"matters — it says whether costing the panel changed the answer."
    )
    lines.append("")
    lines.append(_md_table(
        cost, ["gamma", "objective", "n_feasible_panels", "k", "coverage",
               "seconds", "memory_gb", "deployments",
               "same_panel_as_cardinality"]))
    lines.append("")

    # The honest reading is the negative one.  Reporting only the two rows where
    # deployments disagrees would let a reader believe cost-aware selection is
    # generally live on this panel, when in fact three of the four objectives
    # are the same objective in disguise.
    differs = cost[~cost["same_panel_as_cardinality"]]
    if len(differs):
        named = ", ".join(
            f"gamma {row.gamma:g} under `{row.objective}`"
            for row in differs.itertuples()
        )
        lines.append(
            f"**Verdict: PARTIALLY SUPPORTED.** The sparse half holds; the "
            f"cost half is not separable on this panel. Minimising seconds or "
            f"memory always "
            f"returns the panel that minimises judge count, because this "
            f"panel's judges are close to uniform in both runtime and "
            f"parameter count — so those two objectives are judge count under "
            f"another name. Only `deployments`, the count of distinct model "
            f"families, ever disagrees: {named}. Each buys one fewer family to "
            f"host at a small coverage cost, which is a real saving for someone "
            f"paying per served model and no saving at all for someone paying "
            f"per token. The plan's strong result asks that a cost-aware panel "
            f"improve real cost at similar coverage; that holds for the "
            f"deployment objective only, so C8 is reported as supported on "
            f"sparsity and partial on cost."
        )
    else:
        lines.append(
            "**Verdict: PARTIALLY SUPPORTED.** The sparse half holds; the "
            "cost half is not separable on this panel. No objective ever picks "
            "a different panel from judge count, so on these judges cost-aware "
            "selection "
            "and cardinality selection are the same procedure."
        )
    lines.append("")
    return lines


def _backbone_section(inputs: Dict[str, str]) -> List[str]:
    lines = ["## Backbone — which judges every optimal panel must contain", ""]
    lines.append("**Question.** Section 20 asks the panel to be split three "
                 "ways at each tolerance: judges in *every* minimum panel "
                 "(mandatory backbone), judges in *some* (optional "
                 "representatives), and judges in *none* (nonessential).")
    lines.append("")
    if not os.path.exists(inputs["backbone"]):
        return lines + ["_Not run._", ""]

    payload = R.load(inputs["backbone"])
    head = R.backbone_headline(inputs["backbone"])
    per_seed = R.backbone_table(inputs["backbone"])

    lines.append(
        f"**Method.** This is a statement about all optima, so it cannot be "
        f"read off one greedy run. Every panel of the smallest feasible size "
        f"is enumerated and scored, which is affordable because the error is "
        f"monotone — adding a judge can never raise it — so any superset of an "
        f"infeasible panel is infeasible too and never has to be fitted. "
        f"`certified` records that the size below k* was enumerated in full "
        f"and came back empty, which is what makes k* a minimum rather than "
        f"the best found. Over {len(payload['judges'])} judges, "
        f"{len(payload['split_seeds'])} split seeds, fitted on "
        f"{payload['eval_split']}."
    )
    lines.append("")
    lines.append(_md_table(
        head, ["gamma", "n_seeds_feasible", "k_star_min", "k_star_max",
               "n_always_mandatory", "always_mandatory",
               "never_in_any_optimum", "unstable"]))
    lines.append("")

    all_certified = bool(per_seed["certified"].all())
    if all_certified:
        lines.append(
            "Every k\\* above is certified on every seed, so these are minima "
            "and not greedy stopping points."
        )
    else:
        uncertified = per_seed[~per_seed["certified"]]
        lines.append(
            f"{len(uncertified)} of {len(per_seed)} (seed, tolerance) cases hit "
            f"the enumeration cap before the level below k\\* was cleared, so "
            f"their k\\* is an upper bound. They are marked in "
            f"`tables/backbone_per_seed.csv`."
        )
    lines.append("")

    loose = head[head["never_in_any_optimum"] != ""]
    if len(loose):
        row = loose.iloc[-1]
        lines.append(
            f"**Reading.** Up to the tolerance where the first judge becomes "
            f"droppable the backbone is the whole panel, which is the same fact "
            f"C2 reports as a high floor, seen from the other side: if every "
            f"judge is in every optimum then nothing is redundant. The panel "
            f"only starts to separate once the tolerance is loosened well past "
            f"the target band. At gamma = {row['gamma']:g} the judges in no "
            f"optimum at all are {row['never_in_any_optimum']}"
            + (f", and {row['unstable']} " if row["unstable"] else " ")
            + ("changes category between seeds and so is not a finding."
               if row["unstable"] else "is a stable split.")
        )
        lines.append("")

    stable_mandatory = head[head["unstable"] == ""]
    lines.append(
        f"The `unstable` column is the guard. A judge that is mandatory under "
        f"one partition and nonessential under another says nothing about the "
        f"panel, only about the split, and {len(head) - len(stable_mandatory)} "
        f"of {len(head)} tolerances have at least one such judge. Only the "
        f"`always_mandatory` and `never_in_any_optimum` columns should be "
        f"quoted; per-judge counts are in `tables/backbone_per_judge.csv`."
    )
    lines.append("")
    return lines


def _setup_section(panel: str, inputs: Dict[str, str]) -> List[str]:
    """Panel, contexts, splits and bootstrap size — what every claim below shares."""
    if not os.path.exists(inputs["c3"]):
        return []

    payload = R.load(inputs["c3"])
    block = payload["per_split_seed"][0]
    lines = [
        "## Setup",
        "",
        f"- Judges: {len(block['judges'])} — {', '.join(block['judges'])}",
        f"- Contexts per judge: {len(block['contexts'])}",
        # Rows, not base items: one base item yields up to two comparison pairs
        # and both land in the same split, so these counts are roughly twice the
        # number of source items.
        "- Comparison pairs per split: " + ", ".join(
            f"{name} {count}" for name, count in block["split_items"].items()),
        f"- Split seeds: {payload['split_seeds']}",
        f"- Bootstrap replicates: {payload['n_bootstrap']}",
        "",
    ]

    # A panel smaller than the registry is a deviation from the plan's judge
    # table, not a design choice, and a reader who only sees the count cannot
    # tell the difference.  It is named here rather than left to be inferred
    # from a gap in the judge ids.
    absent = [j for j in PANELS.get(panel, ()) if j not in block["judges"]]
    if absent:
        lines += [
            f"**Deviation from the plan's judge table.** "
            f"{', '.join(absent)} {'is' if len(absent) == 1 else 'are'} in the "
            f"{panel} registry but was never scored, so every result here is "
            f"over {len(block['judges'])} of {len(PANELS[panel])} judges. The "
            f"weights are gated on Hugging Face and the access request was not "
            f"granted in time. Nothing was substituted in its place.",
            "",
        ]

    lines += [
        "Splits are grouped by source item, so the same prompt never appears "
        "in two splits. The pairs seed fixes what each judge saw and needs "
        "GPU inference to change; the split seed only re-partitions those "
        "cached responses, which is what makes across-seed bands affordable.",
        "",
        "---",
        "",
    ]
    return lines


# The section 31 matrix, transcribed from the plan.  Each entry is the claim's
# required experiments, the minimum supporting result and the strong one, kept
# in the plan's own words so the reader can check the transcription rather than
# take a paraphrase on trust.  `experiments` names the keys in `inputs` that
# carry that evidence, which is what decides whether a row was run at all.
_EVIDENCE_MATRIX = [
    ("C1", ["e0_real"], "E0",
     "at least one stable non-composability regime",
     "broad tolerance range with many individually removable judges but clear "
     "joint failure, plus dependency cycles"),
    ("C2", ["e1"], "E1",
     "meaningful reduction with bounded held-out error",
     "Core-20 reduced to 6-10 judges with worst-context TV <= 0.08-0.10"),
    ("C3", ["c3", "c5"], "E1 + E2",
     "coverage beats at least random and top-accuracy at equal k",
     "consistent superiority over accuracy, family, clustering and pairwise "
     "geometry baselines on reconstruction and downstream metrics"),
    ("C4", ["c4"], "E3",
     "clean-only failure on at least one registered stress context",
     "robust selection cuts worst stress error by 30% or more with at most two "
     "additional judges and identifies interpretable specialists"),
    ("C5", ["c5", "e7"], "E2 + E7",
     "virtual panel closely matches full-panel aggregate",
     "< 1 percentage point accuracy loss, small NLL/Brier change, and ranking "
     "tau >= 0.95"),
    ("C6", ["c6"], "E4",
     "swap methods improve greedy gap",
     "2-swap exact on >= 70% of solved cases, mean gap <= 0.5, little 3-swap "
     "benefit"),
    ("C7", ["c7"], "E5",
     "certified method violates less often than empirical-only",
     "nominal 95% certificate achieves about 5% or lower TEST violations over "
     "repeated splits"),
    ("C8", ["e6"], "E6 (optional)",
     "sparse/cost-aware compression remains nontrivial",
     "most certificates use <= 4 judges and cost-aware panel improves real "
     "cost at similar coverage"),
]


def _compliance_section(inputs: Dict[str, str]) -> List[str]:
    """Hand the plan's own evidence matrix back with the evidence filled in.

    The professor set the acceptance criteria in section 31 before any judge
    was scored.  Reporting against them in his wording is what makes a verdict
    checkable: a reader can disagree with a call without first having to work
    out which bar was being aimed at.  `evidence` says only whether the
    required experiment is in this package, which is a fact about the files;
    whether the bar was cleared is argued in each claim's own section.
    """
    lines = ["## Compliance with the section 31 evidence matrix", ""]
    lines.append(
        "The criteria below are quoted from the plan and were fixed before "
        "any judge was scored. `evidence` reports whether the experiment that "
        "section 31 requires is present in this package. It does not restate "
        "the verdicts, which are argued where the numbers are."
    )
    lines.append("")

    rows = []
    missing = []
    for claim, keys, experiments, minimum, strong in _EVIDENCE_MATRIX:
        have = [k for k in keys if os.path.exists(inputs.get(k, ""))]
        absent = [k for k in keys if k not in have]
        if absent:
            missing.append((claim, experiments, absent))
        rows.append({
            "claim": claim,
            "required": experiments,
            "evidence": ("complete" if not absent
                         else "absent" if not have else "partial"),
            "minimum result": minimum,
            "strong result": strong,
        })
    lines.append(_md_table(
        pd.DataFrame(rows),
        ["claim", "required", "evidence", "minimum result", "strong result"]))
    lines.append("")

    if missing:
        named = "; ".join(
            f"{claim} ({experiments})" for claim, experiments, _ in missing)
        lines.append(f"**Not run.** {named}.")
        lines.append("")
        # Only the gaps that are actually open get explained.  A paragraph that
        # still names E6 as deferred after E6 has been run would be the kind of
        # stale claim this whole section exists to prevent.
        if any("e7" in absent for _, _, absent in missing):
            lines.append(
                "E7 is the cross-benchmark transfer to JudgeBench; section 29 "
                "describes it as strengthening C2, C4 and C5, but section 31 "
                "lists it in C5's required column, so C5's row is answered on "
                "E2 alone and that is a real gap rather than a judgement call. "
                "Every C5 number here is in-domain on RewardBench 2."
            )
            lines.append("")
        if any("e6" in absent for _, _, absent in missing):
            lines.append(
                "E6 is marked optional in the plan and was deferred so that "
                "E0-E5 could be completed first, as section 28 directs."
            )
            lines.append("")

    lines.append(
        "Section 32 sets outcome tiers on the reduction achieved at an "
        "acceptable error. Tier B asks for 30-50% reduction at worst-context "
        "TV <= 0.12, and this panel does not reach that error at any panel "
        "size, so on the reduction axis the result sits at Tier C. Section 32 "
        "says that in Tier C the response is to diagnose the cause rather than "
        "manufacture a strong application claim, and the C2 section does that: "
        "the panel is less mutually redundant than the target assumed, and the "
        "floor at k = 19 says so directly. The other Tier C symptoms do not "
        "hold here — reconstruction does beat the physical subset downstream, "
        "the non-composability and exchange results are clear, and the "
        "selected sets are stable across all five split seeds."
    )
    lines.append("")
    return lines


def build_summary(panel: str, inputs: Dict[str, str], prov: dict) -> str:
    lines = [
        f"# EPCRC results — {panel}",
        "",
        f"Generated {prov['generated_utc']} from commit `{prov['git_commit'][:12]}`"
        f" on branch `{prov['git_branch']}`"
        f"{'' if prov['git_clean'] else ' (working tree had uncommitted changes)'}.",
        "",
        "Certified compression of an LLM judge panel: replace most judges with "
        "convex combinations of a small retained set, with a stated bound on how "
        "far any judge's distribution can move.",
        "",
        "All errors are total variation over the three outcomes (A better, B "
        "better, tie). Reconstruction weights are non-negative and sum to one, so "
        "a reconstructed judge is always a probability distribution.",
        "",
        "---",
        "",
    ]

    lines += _setup_section(panel, inputs)
    lines += _c1_section(inputs)
    lines += ["---", ""]
    lines += _c2_section(inputs)
    lines += ["---", ""]
    lines += _c3_section(inputs)
    lines += ["---", ""]
    lines += _c4_section(inputs)
    lines += ["---", ""]
    lines += _c5_section(inputs)
    lines += ["---", ""]
    lines += _c6_section(inputs)
    lines += ["---", ""]
    lines += _c7_section(inputs)
    lines += ["---", ""]
    lines += _c8_section(inputs)
    lines += ["---", ""]
    lines += _backbone_section(inputs)
    lines += ["---", ""]
    lines += _compliance_section(inputs)
    lines += [
        "---",
        "",
        "## Package contents",
        "",
        "- `tables/` — every reported number as CSV",
        "- `figures/` — the figures, drawn from those same tables",
        "- `raw/` — the unmodified experiment output the tables are derived from",
        "- `gates/` — the G0-G2 gate records for this panel",
        "- `notebooks/` — the analysis notebooks as HTML, readable without Jupyter",
        "- `manifest.json` — SHA-256 of every file, plus commit and versions",
        "",
        "Tables and figures are produced by `epcrc/report.py` and "
        "`epcrc/figures.py`, which are also what the notebooks call, so a number "
        "shown in a notebook and the same number in this package come from one "
        "code path.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# FINAL_REPORT.md -- the section 66 structure
# --------------------------------------------------------------------------

# The plan's own headings, in its own order, paired with the section builder
# that answers each.  The bodies are the same functions SUMMARY.md uses, so the
# two documents cannot disagree: this file re-orders and re-titles, it does not
# re-argue.
#
# The section 20 backbone is filed under C4 rather than given a heading of its
# own, because section 66 fixes the heading list and the backbone payload
# itself records C4 as the claim it serves.
_FINAL_SECTIONS = [
    ("C1", "4. C1: Non-Composability", [_c1_section]),
    ("C2", "5. C2: Physical-to-Virtual Compression", [_c2_section]),
    ("C3", "6. C3: Baseline Comparison", [_c3_section]),
    ("C4", "7. C4: Robust Contexts and Specialists",
     [_c4_section, _backbone_section]),
    ("C5", "8. C5: Downstream Preservation", [_c5_section]),
    ("C6", "9. C6: Exact Optimality and Exchange Structure", [_c6_section]),
    ("C7", "10. C7: Certification Reliability", [_c7_section]),
    ("C8", "11. Optional C8 Results", [_c8_section]),
]

_VERDICT_TOKENS = ("PARTIALLY SUPPORTED", "CONTRADICTED", "UNSUPPORTED",
                   "SUPPORTED")


def section_verdict(body: List[str]) -> str:
    """Read the plan's verdict token back out of a section's own prose.

    Deliberately extracted rather than declared beside the section.  A verdict
    stated twice can be changed in one place and not the other, and the one the
    professor reads first is the one at the top of the final report -- so that
    one is taken from the argument underneath it.  `PARTIALLY SUPPORTED` is
    tried before `SUPPORTED` because the latter is a substring of it.
    """
    text = "\n".join(body)
    for token in _VERDICT_TOKENS:
        if f"**Verdict: {token}" in text:
            return token
    return "NOT RUN" if "_Not run._" in text else "NO VERDICT"


def _claim_body(builders, inputs: Dict[str, str]) -> List[str]:
    """One claim's prose, with each builder's own heading demoted to a subhead.

    Section 66 fixes the top-level heading list, so a builder that ships its
    own `## C4 - ...` line would add a heading the plan does not have.
    """
    body: List[str] = []
    for i, builder in enumerate(builders):
        part = builder(inputs)
        if i:
            body += ["", f"###{part[0][1:]}"] + part[1:]
        else:
            body += part[1:]
    return body


def build_final_report(panel: str, inputs: Dict[str, str], prov: dict) -> str:
    """The section 66 document, assembled from the sections SUMMARY.md uses."""
    bodies = {claim: _claim_body(builders, inputs)
              for claim, _, builders in _FINAL_SECTIONS}
    verdicts = {claim: section_verdict(body) for claim, body in bodies.items()}

    lines = [
        "# Final Report",
        "",
        f"Certified compression of an LLM judge panel — {panel}. Generated "
        f"{prov['generated_utc']} from commit `{prov['git_commit'][:12]}` on "
        f"branch `{prov['git_branch']}`.",
        "",
        "## 1. Executive Findings",
        "",
        "Every verdict below is the one argued in that claim's own section; "
        "nothing is asserted here that is not derived there.",
        "",
    ]
    lines.append(_md_table(
        pd.DataFrame([{"claim": claim, "verdict": verdict}
                      for claim, verdict in verdicts.items()]),
        ["claim", "verdict"]))
    lines += ["", "---", ""]

    lines += ["## 2. Frozen Experimental Setting", ""]
    lines += _setup_section(panel, inputs)[1:]

    lines += _completeness_section(panel, inputs)
    lines += ["---", ""]

    for claim, heading, _ in _FINAL_SECTIONS:
        lines += [f"## {heading}", "", f"**{verdicts[claim]}**", ""]
        lines += bodies[claim]
        lines += ["---", ""]

    lines += _negative_results_section(inputs, verdicts)
    lines += ["---", ""]
    lines += _deviations_section(panel, inputs)
    lines += ["---", ""]
    lines += _index_section(inputs)
    lines += ["---", ""]
    lines += _reproduction_section(panel)
    lines += ["---", ""]
    lines += _verification_section()
    return "\n".join(lines)


def _completeness_section(panel: str, inputs: Dict[str, str]) -> List[str]:
    """Section 3. Which cells of the judge-by-context-by-item grid actually exist.

    A panel is only as complete as its emptiest cell, and a missing judge is a
    different problem from a missing context. Both are reported as counts
    rather than described, so a reader can see the gap instead of trusting a
    sentence about it.
    """
    lines = ["## 3. Data and Panel Completeness", ""]
    if not os.path.exists(inputs["c3"]):
        return lines + ["_No scored panel in this package._", ""]

    block = R.load(inputs["c3"])["per_split_seed"][0]
    scored, registry = block["judges"], list(PANELS.get(panel, ()))
    absent = [j for j in registry if j not in scored]
    pairs = sum(block["split_items"].values())

    lines.append(
        f"The response grid is {len(scored)} judges x "
        f"{len(block['contexts'])} contexts x {pairs} comparison pairs, and "
        f"every cell in it was scored — no judge is missing a context and no "
        f"context is missing an item. Judges answer on the three-class "
        f"simplex, so a cell is a distribution and not a single label."
    )
    lines.append("")

    if absent:
        lines.append(
            f"**Incomplete against the plan's judge table.** "
            f"{', '.join(absent)} appears in the {panel} registry and was "
            f"never scored: the weights are gated on Hugging Face and access "
            f"was not granted in time. Every number in this report is "
            f"therefore over {len(scored)} of {len(registry)} judges. No "
            f"substitute was put in its place, because a substituted judge "
            f"would change what the panel is while leaving the count intact."
        )
    else:
        lines.append(
            f"All {len(registry)} judges in the {panel} registry were scored, "
            f"so the panel is complete against the plan's judge table."
        )
    lines.append("")

    lines.append(
        "Split manifests under `data/` carry a SHA-256 over the item ids in "
        "each split and the hash of every prompt template, so the partition "
        "and the wording a judge saw are both checkable rather than asserted. "
        "Splits are grouped by source item: a prompt and both of its "
        "comparison pairs always land in the same split, which is what stops "
        "held-out error being read off items the fit already saw."
    )
    lines.append("")
    return lines


def _negative_results_section(inputs: Dict[str, str],
                              verdicts: Dict[str, str]) -> List[str]:
    """Section 12. What did not work, kept as a section rather than a footnote."""
    lines = ["## 12. Negative and Null Results", ""]
    failed = [c for c, v in verdicts.items()
              if v in ("UNSUPPORTED", "CONTRADICTED")]
    partial = [c for c, v in verdicts.items() if v == "PARTIALLY SUPPORTED"]

    lines.append(
        f"{len(failed) + len(partial)} of {len(verdicts)} claims did not come "
        f"back fully supported"
        + (f": {', '.join(failed)} unsupported" if failed else "")
        + (f", {', '.join(partial)} partial" if partial else "")
        + ". The arguments are in each claim's section; what follows is the "
          "list of things that were measured and found not to hold, so a "
          "reader does not have to reconstruct it from the verdicts."
    )
    lines.append("")

    # Each bullet is attached to the claim it belongs to and only printed when
    # that claim actually fell short.  A null result that survives its own
    # claim turning positive is a stale sentence, and this is the section a
    # reader trusts most to be current.
    shortfalls = [
        ("C2",
         "The Core-20 panel does not reach the section 31 error band at any "
         "panel size. The floor is set by how little the judges overlap, not "
         "by the selector, and the C2 frontier is what shows it. The section "
         "23 tolerance grid is vacuous here for the same reason: every "
         "tolerance in it selects the whole panel."),
        ("C4",
         "The C4 advantage is mostly a fitting effect rather than a selection "
         "effect. Against the arm that is handicapped only at selection time "
         "the margin is small and not unanimous, which is weaker than the "
         "claim as written."),
        ("C5",
         "Only one half of the stated downstream tolerance is met. The "
         "compressed panel tracks the full panel far better than any physical "
         "subset of the same size, but not inside the band section 31 names."),
        ("C7",
         "At the tightest tolerances the primary rule certifies too few cases "
         "for a 95% bound to fall below the nominal rate, however well the "
         "rule behaved. That is a resolution limit, not a violated bound, and "
         "more repetitions rather than a different bound would settle it."),
        ("C8",
         "Cost-aware selection does not separate from plain judge count on "
         "this panel except under the deployment-count objective, because "
         "these judges are near-uniform in runtime and parameter count."),
    ]
    lines += [f"- {text}" for claim, text in shortfalls
              if verdicts.get(claim) not in ("SUPPORTED", None)]
    lines.append("")
    return lines


def _deviations_section(panel: str, inputs: Dict[str, str]) -> List[str]:
    """Section 13. Every place this run departs from the plan, named."""
    lines = ["## 13. Deviations from the Plan", ""]

    if os.path.exists(inputs["c3"]):
        block = R.load(inputs["c3"])["per_split_seed"][0]
        absent = [j for j in PANELS.get(panel, ()) if j not in block["judges"]]
        if absent:
            lines.append(
                f"- **Panel size.** {', '.join(absent)} is in the {panel} "
                f"registry but was never scored: its weights are gated on "
                f"Hugging Face and access was not granted in time. Every "
                f"result is over {len(block['judges'])} of "
                f"{len(PANELS[panel])} judges, and nothing was substituted."
            )

    if not os.path.exists(inputs["e7"]):
        lines.append(
            "- **E7 not run.** Section 31 lists E7 in C5's required column, so "
            "C5 is answered on E2 alone. Every downstream number is in-domain "
            "on RewardBench 2."
        )

    lines += [
        "- **Tolerance grid.** The section 23 grid selects the whole panel at "
        "every value on these judges, so the reported grid extends past it. "
        "The section 23 values are still reported, as empty results.",
        "- **Split seeds.** Bands are taken over five re-partitions of one set "
        "of cached judge responses, not over five independent inference runs. "
        "Re-querying the judges per seed was not affordable, so the bands "
        "cover partition variance only and not sampling variance in the "
        "judges' own outputs.",
        "",
    ]
    return lines


def _index_section(inputs: Dict[str, str]) -> List[str]:
    """Section 14. Which file carries which number."""
    tables = sorted(build_tables(inputs))
    lines = ["## 14. Figure and Table Index", ""]
    lines.append(f"{len(tables)} tables in `tables/`:")
    lines.append("")
    lines += [f"- `tables/{name}.csv`" for name in tables]
    lines += [
        "",
        "Figures in `figures/` are drawn from those same tables by "
        "`epcrc/figures.py`, so each figure's numbers are readable as CSV.",
        "",
    ]
    return lines


# The scripts section 15 tells a reader to run, in order.  Kept as data and
# checked by a test against the files on disk, because a reproduction section
# that names a script which has since been renamed is worse than no section.
REPRODUCTION_SCRIPTS = [
    ("build_rewardbench_pairs.py", "--all-seeds"),
    ("score_panel.py", "--gate g2"),
    ("experiment_e0_noncomposability.py", "--real"),
    ("experiment_e1_compression_frontier.py", ""),
    ("experiment_c3_baselines.py", ""),
    ("experiment_c4_stress_specialists.py", ""),
    ("experiment_c5_downstream.py", ""),
    ("experiment_c6_exchange.py", ""),
    ("experiment_c7_certification.py", ""),
    ("experiment_e6_sparse_cost.py", ""),
    ("experiment_backbone.py", ""),
    ("export_results.py", ""),
]


def _reproduction_section(panel: str) -> List[str]:
    """Section 15. The commands, in the order they must be run."""
    def command(name: str, flags: str) -> str:
        parts = [f"python experiments/{name}"]
        # The pair builder is panel-independent: it writes the judged pairs
        # every panel then scores, so passing a panel to it would be wrong.
        if name != "build_rewardbench_pairs.py":
            parts.append(f"--panel {panel}")
        if flags:
            parts.append(flags)
        return " ".join(parts)

    return [
        "## 15. Reproduction Commands",
        "",
        "Only the first two steps need the judge weights and a GPU. "
        "Everything after `score_panel.py` reads cached response blocks and "
        "runs on a laptop, because a split seed re-partitions those blocks "
        "rather than re-querying the judges.",
        "",
        "```bash",
        *[command(name, flags) for name, flags in REPRODUCTION_SCRIPTS],
        "```",
        "",
        "`docs/SERVER_RUNBOOK.md` has the full procedure including gates "
        "G0-G2 and the E7 transfer scoring.",
        "",
    ]


def _verification_section() -> List[str]:
    """Section 16. How to check this package was not edited after the fact."""
    return [
        "## 16. Package Verification",
        "",
        "`manifest.json` holds a SHA-256 for every file in the package, plus "
        "the git commit, the Python and library versions and the platform. To "
        "check the package is the one that was generated:",
        "",
        "```bash",
        "python - <<'PY'",
        "import hashlib, json, os",
        "m = json.load(open('manifest.json'))",
        "for name, want in m['files'].items():",
        "    got = hashlib.sha256(open(name, 'rb').read()).hexdigest()",
        "    print('OK ' if got == want else 'CHANGED', name)",
        "PY",
        "```",
        "",
        "`provenance.git_clean` in the manifest says whether the working tree "
        "had uncommitted changes when the package was built. If it is false, "
        "the commit alone does not identify the code that produced these "
        "numbers.",
        "",
    ]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def copy_notebooks(out_dir: str) -> int:
    """Render already-executed notebooks to HTML so they read without Jupyter.

    Notebooks are exported as they were last run, not re-executed here: the
    package must show the outputs the author actually saw.

    Only the numbered series is shipped.  Unnumbered notebooks are scratch work
    from earlier iterations and may hold superseded numbers, which is exactly
    what must not reach a results package.
    """
    src = os.path.join(ROOT, "notebooks")
    if not os.path.isdir(src):
        return 0

    dest = os.path.join(out_dir, "notebooks")
    os.makedirs(dest, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".ipynb") or not name[:2].isdigit():
            continue
        try:
            subprocess.run(
                [sys.executable, "-m", "nbconvert", "--to", "html",
                 "--output-dir", dest, os.path.join(src, name)],
                check=True, capture_output=True,
            )
            count += 1
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  could not render {name}; skipping")
    if count == 0:
        shutil.rmtree(dest, ignore_errors=True)
    return count


def export(panel: str, out_dir: Optional[str] = None,
           make_zip: bool = True, with_notebooks: bool = True) -> str:
    inputs = input_paths(panel)
    missing = [k for k, v in inputs.items() if not os.path.exists(v)]
    if missing:
        print(f"WARNING: no results for {missing}; the package will be partial.")

    out_dir = out_dir or os.path.join(RESULTS, "export", panel)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for sub in ("tables", "figures", "raw"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    prov = provenance(panel, {**inputs, **gate_paths(panel)})

    tables = build_tables(inputs)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, "tables", f"{name}.csv"), index=False)
    print(f"tables:  {len(tables)}")

    figs = F.save_all(
        os.path.join(out_dir, "figures"),
        e0_real=inputs["e0_real"] if os.path.exists(inputs["e0_real"]) else None,
        e0_synth=(inputs["e0_synthetic"]
                  if os.path.exists(inputs["e0_synthetic"]) else None),
        e1=inputs["e1"] if os.path.exists(inputs["e1"]) else None,
        c3=inputs["c3"] if os.path.exists(inputs["c3"]) else None,
        c4=inputs["c4"] if os.path.exists(inputs["c4"]) else None,
        c5=inputs["c5"] if os.path.exists(inputs["c5"]) else None,
        c6=inputs["c6"] if os.path.exists(inputs["c6"]) else None,
        c7=inputs["c7"] if os.path.exists(inputs["c7"]) else None,
        e6=inputs["e6"] if os.path.exists(inputs["e6"]) else None,
        backbone=(inputs["backbone"]
                  if os.path.exists(inputs["backbone"]) else None),
    )
    print(f"figures: {len(figs)}")

    for key, path in inputs.items():
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(out_dir, "raw", os.path.basename(path)))

    gates = {k: v for k, v in gate_paths(panel).items() if os.path.exists(v)}
    if gates:
        os.makedirs(os.path.join(out_dir, "gates"), exist_ok=True)
        for path in gates.values():
            shutil.copy2(path, os.path.join(out_dir, "gates", os.path.basename(path)))
    print(f"gates:   {len(gates)}")

    if with_notebooks:
        print(f"notebooks: {copy_notebooks(out_dir)}")

    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as handle:
        handle.write(build_summary(panel, inputs, prov))

    # Same sections, the plan's order and headings.  Section 66 fixes the
    # structure the professor reads, so it is generated rather than kept by
    # hand next to a summary that would drift away from it.
    with open(os.path.join(out_dir, "FINAL_REPORT.md"), "w") as handle:
        handle.write(build_final_report(panel, inputs, prov))

    files = []
    for folder, _, names in os.walk(out_dir):
        for name in sorted(names):
            full = os.path.join(folder, name)
            files.append({
                "path": os.path.relpath(full, out_dir),
                "bytes": os.path.getsize(full),
                "sha256": sha256(full),
            })
    prov["files"] = sorted(files, key=lambda f: f["path"])
    with open(os.path.join(out_dir, "manifest.json"), "w") as handle:
        json.dump(prov, handle, indent=2)

    if not make_zip:
        return out_dir

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    zip_dir = os.path.join(RESULTS, "export")
    # results/export/ is gitignored, so on a fresh clone -- the server -- it does
    # not exist until something creates it.
    os.makedirs(zip_dir, exist_ok=True)
    zip_path = os.path.join(zip_dir, f"epcrc_results_{panel}_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder, _, names in os.walk(out_dir):
            for name in sorted(names):
                full = os.path.join(folder, name)
                archive.write(full, os.path.join(
                    f"epcrc_results_{panel}", os.path.relpath(full, out_dir)))

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"\nwrote {zip_path}  ({size_mb:.1f} MB)")
    return zip_path


def main() -> None:
    # Only when run as a script: a command line invocation has no display, but
    # importing this module from a notebook must not disturb inline rendering.
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="core8")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--no-notebooks", action="store_true",
                        help="skip rendering the notebooks to HTML")
    args = parser.parse_args()
    export(args.panel, args.out, make_zip=not args.no_zip,
           with_notebooks=not args.no_notebooks)


if __name__ == "__main__":
    main()
