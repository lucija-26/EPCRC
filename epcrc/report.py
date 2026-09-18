"""Turn the raw experiment JSON into the tables that go in the paper.

This module is the single source of truth for every reported number.  The
notebooks call it to display results and the export script calls it to write
the CSVs, so a figure in a notebook and a row in the delivered zip can never
disagree -- which is the failure mode that makes a results package impossible
to defend.

Nothing here recomputes an experiment.  If a number is wrong, it is wrong in
the JSON and the experiment has to be rerun.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "load",
    "c1_table",
    "c1_headline",
    "c2_table",
    "c3_table",
    "c3_headline",
    "c3_paired_table",
    "c3_split_robustness",
    "c3_cost_table",
    "c4_table",
    "c4_headline",
    "c4_specialists",
    "c5_table",
    "c5_headline",
    "c6_table",
    "c6_headline",
    "c7_table",
    "c7_headline",
    "reconstruction_table",
    "lowrank_floor_table",
    "fmt_ci",
]

# The method under test, and the baselines it has to beat.  Order is the order
# they appear in the paper table.
COVERAGE_METHODS = ["coverage_backward", "coverage_forward", "exhaustive"]
BASELINE_ORDER = [
    "top_accuracy",
    "random",
    "one_per_family",
    "cost_ascending",
    "cost_descending",
    "correlation_medoid",
    "kmedoids",
    "hierarchical",
    "farthest_first",
    "pivoted_qr",
    "leverage",
]

PRETTY = {
    "coverage_backward": "Coverage (backward)",
    "coverage_forward": "Coverage (forward)",
    "exhaustive": "Coverage (exact)",
    "top_accuracy": "Top accuracy",
    "random": "Random",
    "random_best_draw": "Random (luckiest draw)",
    "one_per_family": "One per family",
    "cost_ascending": "Cheapest first",
    "cost_descending": "Largest first",
    "correlation_medoid": "Correlation clustering",
    "kmedoids": "k-medoids",
    "hierarchical": "Hierarchical",
    "farthest_first": "Farthest first",
    "pivoted_qr": "Pivoted QR",
    "leverage": "Leverage scores",
    "pca": "PCA floor (not deployable)",
    "nmf": "NMF floor (not deployable)",
}


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def fmt_ci(value: float, lo: float, hi: float, digits: int = 3) -> str:
    """The paper's number format: point estimate with a bracketed interval."""
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _family(method: str) -> str:
    """Collapse the 100 random draws into a single reported baseline."""
    return "random" if method.startswith("random_") else method


# --------------------------------------------------------------------------
# C1 -- non-composability
# --------------------------------------------------------------------------

def c1_table(path: str) -> pd.DataFrame:
    """One row per (instance, gamma, split seed) from E0."""
    payload = load(path)
    rows = []
    for r in payload["records"]:
        rows.append({
            "instance": r["instance"],
            "gamma": r["gamma"],
            "gamma_source": r.get("gamma_source", "declared"),
            "seed": r.get("seed", r.get("split_seed")),
            "n_individually_removable": r["n_individually_removable"],
            "naive_retained_size": r["naive_retained_size"],
            "naive_coverage": r["naive_coverage"],
            "naive_violates_gamma": r["naive_violates_gamma"],
            "min_feasible_size": r["min_feasible_size"],
            "min_feasible_is_exact": r["min_feasible_is_exact"],
            "min_feasible_lower_bound": r["min_feasible_lower_bound"],
            "composition_gap": r["composition_gap"],
            "n_cycles": r["dependency_graph"]["n_cycles"],
            "loo_min": float(np.min(r["loo_errors"])),
            "loo_median": float(np.median(r["loo_errors"])),
            "loo_max": float(np.max(r["loo_errors"])),
        })
    return pd.DataFrame(rows)


def c1_headline(path: str, gamma_source: Optional[str] = "declared") -> pd.DataFrame:
    """Mean over seeds of the quantities C1 is actually about.

    ``naive_coverage > gamma`` is the claim: every judge passed its own
    leave-one-out test, yet deleting them together breaks the tolerance.

    `gamma_source` selects the tolerance grid: "declared" is the predeclared
    section 23 grid, "loo_breakpoint" is the data-driven diagnostic, and None
    pools both.  On a panel whose leave-one-out errors all sit above the
    declared grid, only the breakpoints carry any signal.

    `min_feasible_exact` says whether `min_feasible` and `composition_gap` are
    the true minimum or only backward elimination's upper bound on it, which
    happens once the panel outgrows E0's exhaustive budget.  Where it is False
    the gap is a bound and its sign proves nothing, so the two columns must be
    read as ``<=``.  The claim itself never depends on this: `naive_set_fails`
    is one coverage evaluation and is exact on every row.
    """
    df = c1_table(path)
    if gamma_source is not None:
        df = df[df["gamma_source"] == gamma_source]
    out = (
        df.groupby(["instance", "gamma"])
        .agg(
            removable=("n_individually_removable", "mean"),
            naive_size=("naive_retained_size", "mean"),
            naive_coverage=("naive_coverage", "mean"),
            min_feasible=("min_feasible_size", "mean"),
            min_feasible_exact=("min_feasible_is_exact", "all"),
            min_feasible_floor=("min_feasible_lower_bound", "mean"),
            composition_gap=("composition_gap", "mean"),
            cycles=("n_cycles", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    out["naive_set_fails"] = out["naive_coverage"] > out["gamma"]
    return out


# --------------------------------------------------------------------------
# C2 -- the compression frontier
# --------------------------------------------------------------------------

def c2_table(path: str, split: str = "TEST") -> pd.DataFrame:
    """One row per (method, k) from E1, on the locked split."""
    payload = load(path)
    rows = []
    for method, body in payload["methods"].items():
        for k, row in body[split].items():
            rows.append({
                "method": _family(method),
                "raw_method": method,
                "k": int(k),
                "worst_judge_tv": row["worst_judge_worst_context_tv"],
                "mean_judge_tv": row["mean_judge_worst_context_tv"],
                "median_item_tv": row["median_item_tv"],
                "p95_item_tv": row["p95_item_tv"],
                "verdict_agreement": row["verdict_agreement"],
                "calls_avoided_frac": row["calls_avoided_frac"],
            })
    return pd.DataFrame(rows).sort_values(["method", "k"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# C3 -- the baseline comparison
# --------------------------------------------------------------------------

def c3_table(path: str, split: str = "TEST") -> pd.DataFrame:
    """One row per (method, k), averaged over split seeds.

    The interval is the width of the item bootstrap, averaged across seeds,
    recentred on the across-seed mean.  Random draws are pooled into a single
    `random` baseline because the plan asks for the distribution over draws,
    not for 100 separate competitors.
    """
    payload = load(path)
    records = []
    for block in payload["per_split_seed"]:
        for method, body in block["methods"].items():
            for k, row in body.items():
                ci = row.get(f"{split}_ci", {}).get("worst_judge_worst_context_tv", {})
                records.append({
                    "method": _family(method),
                    "k": int(k),
                    "split_seed": block["split_seed"],
                    "worst_judge_tv": row[split]["worst_judge_worst_context_tv"],
                    "mean_judge_tv": row[split]["mean_judge_worst_context_tv"],
                    "verdict_agreement": row[split]["verdict_agreement"],
                    "p95_item_tv": row[split]["p95_item_tv"],
                    "cost_frac": row["cost_frac"],
                    "calls_avoided_frac": row["calls_avoided_frac"],
                    "ci_lo": ci.get("lo", np.nan),
                    "ci_hi": ci.get("hi", np.nan),
                })

    df = pd.DataFrame(records)
    out = (
        df.groupby(["method", "k"])
        .agg(
            worst_judge_tv=("worst_judge_tv", "mean"),
            worst_judge_tv_sd=("worst_judge_tv", "std"),
            worst_judge_tv_min=("worst_judge_tv", "min"),
            worst_judge_tv_max=("worst_judge_tv", "max"),
            mean_judge_tv=("mean_judge_tv", "mean"),
            verdict_agreement=("verdict_agreement", "mean"),
            p95_item_tv=("p95_item_tv", "mean"),
            cost_frac=("cost_frac", "mean"),
            calls_avoided_frac=("calls_avoided_frac", "mean"),
            boot_lo=("ci_lo", "mean"),
            boot_hi=("ci_hi", "mean"),
            n_observations=("worst_judge_tv", "size"),
        )
        .reset_index()
    )
    out["label"] = out["method"].map(lambda m: PRETTY.get(m, m))
    return out.sort_values(["method", "k"]).reset_index(drop=True)


def c3_headline(
    path: str,
    k_values: Optional[List[int]] = None,
    split: str = "TEST",
) -> pd.DataFrame:
    """The head-to-head table: one row per method, averaged over panel sizes.

    Trivial budgets are excluded by default.  k = N reconstructs every judge by
    itself at zero error for every method, and k = 1 admits no convex
    combination at all, so including either would dilute the comparison with
    sizes at which no method can differ.
    """
    df = c3_table(path, split)
    sizes = sorted(int(k) for k in df["k"].unique())
    if k_values is None:
        k_values = [k for k in sizes if 1 < k < max(sizes)]

    sub = df[df["k"].isin(k_values)]
    out = (
        sub.groupby("method")
        .agg(
            worst_judge_tv=("worst_judge_tv", "mean"),
            mean_judge_tv=("mean_judge_tv", "mean"),
            verdict_agreement=("verdict_agreement", "mean"),
            p95_item_tv=("p95_item_tv", "mean"),
            boot_lo=("boot_lo", "mean"),
            boot_hi=("boot_hi", "mean"),
        )
        .reset_index()
    )

    order = [m for m in COVERAGE_METHODS + BASELINE_ORDER if m in set(out["method"])]
    out["rank_order"] = out["method"].map({m: i for i, m in enumerate(order)})
    out = out.sort_values("rank_order").drop(columns="rank_order")
    out["label"] = out["method"].map(lambda m: PRETTY.get(m, m))
    out["is_coverage"] = out["method"].isin(COVERAGE_METHODS)
    out["k_values"] = [list(k_values)] * len(out)

    best = out["worst_judge_tv"].min()
    out["best"] = np.isclose(out["worst_judge_tv"], best)
    return out.reset_index(drop=True)


def c3_paired_table(path: str, reference: str = "coverage_backward") -> pd.DataFrame:
    """Section 34.2: the paired bootstrap of the reference against each baseline.

    `delta` is ``reference - baseline`` on the *same* resampled items, so a
    negative value means the reference wins.  This is the sensitive test:
    checking whether two independent intervals happen to overlap throws away
    the fact that both methods were scored on one set of items, so most of the
    sampling noise is shared and cancels in the difference.

    Averaged over split seeds; `n_seeds_significant` counts how many of them
    put the whole interval on one side of zero.
    """
    payload = load(path)
    rows = []
    for block in payload["per_split_seed"]:
        for k, body in block.get("paired_TEST", {}).items():
            for method, row in body.items():
                rows.append({
                    "method": method,
                    "k": int(k),
                    "split_seed": block["split_seed"],
                    "delta": row["delta"],
                    "lo": row["lo"],
                    "hi": row["hi"],
                    "prob_reference_better": row["prob_a_better"],
                    "excludes_zero": row["excludes_zero"],
                })
    if not rows:
        return pd.DataFrame(columns=[
            "method", "label", "k", "delta", "lo", "hi",
            "prob_reference_better", "n_seeds_significant", "n_seeds",
        ])

    df = pd.DataFrame(rows)
    out = (
        df.groupby(["method", "k"])
        .agg(
            delta=("delta", "mean"),
            lo=("lo", "mean"),
            hi=("hi", "mean"),
            prob_reference_better=("prob_reference_better", "mean"),
            n_seeds_significant=("excludes_zero", "sum"),
            n_seeds=("excludes_zero", "size"),
        )
        .reset_index()
    )
    out["label"] = out["method"].map(lambda m: PRETTY.get(m, m))
    out["reference"] = reference
    # A baseline that happens to pick the identical subset gives delta exactly
    # zero.  That is a tie, not a loss, so it is labelled as one.
    out["outcome"] = np.where(
        np.isclose(out["delta"], 0.0), "tie",
        np.where(out["delta"] < 0, "reference better", "baseline better"),
    )
    order = {m: i for i, m in enumerate(COVERAGE_METHODS + BASELINE_ORDER)}
    out["rank_order"] = out["method"].map(lambda m: order.get(_family(m), len(order)))
    return (
        out.sort_values(["k", "rank_order", "method"])
        .drop(columns="rank_order")
        .reset_index(drop=True)
    )


def c3_split_robustness(
    path: str,
    k_values: Optional[List[int]] = None,
    split: str = "TEST",
) -> pd.DataFrame:
    """Section 34.4: mean, SD, min and max across the five split seeds.

    Each seed contributes one number per method -- its worst-judge error
    averaged over the non-trivial budgets -- and the spread of those numbers is
    what shows the headline is not resting on one favourable partition.
    """
    payload = load(path)
    records = []
    for block in payload["per_split_seed"]:
        for method, body in block["methods"].items():
            for k, row in body.items():
                records.append({
                    "method": _family(method),
                    "k": int(k),
                    "split_seed": block["split_seed"],
                    "worst_judge_tv": row[split]["worst_judge_worst_context_tv"],
                })
    df = pd.DataFrame(records)

    sizes = sorted(df["k"].unique())
    if k_values is None:
        k_values = [k for k in sizes if 1 < k < max(sizes)]
    df = df[df["k"].isin(k_values)]

    per_seed = (
        df.groupby(["method", "split_seed"])["worst_judge_tv"].mean().reset_index()
    )
    out = (
        per_seed.groupby("method")["worst_judge_tv"]
        .agg(mean="mean", sd="std", min="min", max="max", n_seeds="size")
        .reset_index()
    )
    out["range"] = out["max"] - out["min"]
    out["label"] = out["method"].map(lambda m: PRETTY.get(m, m))
    out["k_values"] = [list(k_values)] * len(out)

    order = [m for m in COVERAGE_METHODS + BASELINE_ORDER if m in set(out["method"])]
    out["rank_order"] = out["method"].map({m: i for i, m in enumerate(order)})
    return (
        out.sort_values("rank_order").drop(columns="rank_order").reset_index(drop=True)
    )


def c3_cost_table(path: str, split: str = "TEST") -> pd.DataFrame:
    """Cost side of C3: what each panel size actually saves.

    The plan asks for comparison at equal cost as well as at equal k, and
    judges are not equally expensive, so the honest efficiency claim is stated
    in measured inference seconds rather than in judge count.
    """
    payload = load(path)
    first = payload["per_split_seed"][0]
    total = sum(first["cost_seconds"].values())

    df = c3_table(path, split)
    cov = df[df["method"] == "coverage_backward"].copy()
    cov["panel_seconds"] = cov["cost_frac"] * total
    cov["seconds_saved"] = total - cov["panel_seconds"]
    cov["speedup"] = total / cov["panel_seconds"].replace(0, np.nan)
    return cov[[
        "k", "worst_judge_tv", "verdict_agreement", "calls_avoided_frac",
        "panel_seconds", "seconds_saved", "speedup",
    ]].reset_index(drop=True)


def lowrank_floor_table(path: str) -> pd.DataFrame:
    """The nondeployable rank-k bound, averaged over split seeds."""
    payload = load(path)
    rows = []
    for block in payload["per_split_seed"]:
        for kind, table in block["lowrank_floor_TEST"].items():
            for k, row in table.items():
                rows.append({
                    "method": kind,
                    "k": int(k),
                    "split_seed": block["split_seed"],
                    "worst_judge_tv": row["worst_judge_worst_context_tv"],
                    "mean_judge_tv": row["mean_judge_worst_context_tv"],
                })
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["method", "k"])
        .agg(worst_judge_tv=("worst_judge_tv", "mean"),
             mean_judge_tv=("mean_judge_tv", "mean"))
        .reset_index()
    )
    out["label"] = out["method"].map(lambda m: PRETTY.get(m, m))
    return out


# --------------------------------------------------------------------------
# C4 -- stress specialists
# --------------------------------------------------------------------------

def c4_table(path: str) -> pd.DataFrame:
    """One row per (split seed, arm, budget) from C4, on the locked TEST split.

    Every arm is scored on all seven contexts however few it was allowed to
    select on, so `worst_context_tv` is comparable across arms at equal `k`.
    """
    payload = load(path)
    rows = []
    for seed, block in payload["per_seed"].items():
        for arm, body in block["arms"].items():
            for k, row in body["budgets"].items():
                rows.append({
                    "split_seed": int(seed),
                    "arm": arm,
                    "selects_on": ",".join(body["selects_on"]),
                    "fits_on": ",".join(body["fits_on"]),
                    "k": int(k),
                    "worst_context_tv": row["worst_judge_worst_context_tv"],
                    "clean_tv": row["worst_judge_clean_tv"],
                    "mean_judge_tv": row["mean_judge_worst_context_tv"],
                    "verdict_agreement": row["verdict_agreement"],
                    "n_specialists_kept": row["n_specialists_kept"],
                    "n_specialists_total": row["n_specialists_total"],
                    "kept": ",".join(row["kept"]),
                })
    return (
        pd.DataFrame(rows)
        .sort_values(["arm", "k", "split_seed"])
        .reset_index(drop=True)
    )


def c4_headline(path: str) -> pd.DataFrame:
    """What C4 turns on: robust selection minus each clean-only baseline.

    `delta` is baseline minus robust, so a positive value is the direction the
    claim predicts.  It is reported with the across-seed spread and with
    `n_seeds_better` because a mean delta that rests on one favourable partition
    is not evidence; the two have to be read together.
    """
    payload = load(path)
    rows = []
    for key, table in payload["summary"].items():
        if not key.startswith("robust_vs_"):
            continue
        for k, row in table.items():
            rows.append({
                "baseline": key[len("robust_vs_"):],
                "k": int(k),
                "delta_mean": row["delta_worst_context_mean"],
                "delta_sd": row["delta_worst_context_sd"],
                "delta_min": row["delta_worst_context_min"],
                "delta_max": row["delta_worst_context_max"],
                "n_seeds_better": row["n_seeds_robust_better"],
                "n_seeds": row["n_seeds"],
                "specialist_advantage": row["specialists_retained_advantage_mean"],
            })
    out = pd.DataFrame(rows)
    # Unanimity across seeds is the bar, not a positive mean: at five seeds a
    # 4/5 split is a visible caveat and has to survive into the table.
    out["robust_wins_every_seed"] = out["n_seeds_better"] == out["n_seeds"]
    return out.sort_values(["baseline", "k"]).reset_index(drop=True)


def c4_specialists(path: str) -> pd.DataFrame:
    """Which judges are stress specialists, and how stable that label is.

    A judge identified only under one partition is a candidate, not a finding,
    so `in_every_seed` is what the text should quote.
    """
    payload = load(path)
    stability = payload["summary"]["specialist_stability"]
    always = set(stability["specialists_in_every_seed"])

    seen: Dict[str, List[str]] = {}
    for seed, block in payload["per_seed"].items():
        for judge, body in block["specialists"].items():
            if body["is_specialist"]:
                seen.setdefault(judge, []).append(str(seed))

    rows = [
        {
            "judge": judge,
            "n_seeds_flagged": len(seeds),
            "in_every_seed": judge in always,
            "binding_contexts": ",".join(stability["binding_contexts"].get(judge, [])),
        }
        for judge, seeds in sorted(seen.items())
    ]
    # A panel with no stress specialists at all is a legitimate outcome -- it is
    # C4 failing -- so the columns are declared rather than inferred from `rows`.
    # Otherwise that outcome yields a frame with no columns and every consumer
    # raises KeyError on the one result it most needs to be able to report.
    return pd.DataFrame(
        rows,
        columns=["judge", "n_seeds_flagged", "in_every_seed", "binding_contexts"],
    )


# --------------------------------------------------------------------------
# C5 -- downstream preservation
# --------------------------------------------------------------------------

# The comparison C5 turns on.  `physical` is the same budget without
# reconstruction, so the gap between it and `virtual` is what reconstruction
# buys; everything else is a heuristic panel of the same size.
C5_ARM_ORDER = [
    "full", "virtual", "physical",
    "top_accuracy", "one_per_family", "random", "single_best",
]

C5_PRETTY = {
    "full": "Full panel (reference)",
    "virtual": "Virtual (k real + reconstructions)",
    "physical": "Physical (k real only)",
    "top_accuracy": "Top accuracy",
    "one_per_family": "One per family",
    "random": "Random",
    "single_best": "Single best judge",
}


def c5_table(path: str) -> pd.DataFrame:
    """One row per (split seed, arm, budget, context, aggregator) from E2."""
    payload = load(path)
    rows = []
    for seed, block in payload["per_seed"].items():
        for key, record in block.items():
            for context_key, metrics in record["contexts"].items():
                context, aggregator = context_key.split("|")
                rows.append({
                    "split_seed": int(seed),
                    "arm": record.get("arm", key),
                    "k": record["k"],
                    "context": context,
                    "aggregator": aggregator,
                    "kept": ",".join(record["kept"]),
                    **{
                        name: value for name, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                })
    return (
        pd.DataFrame(rows)
        .sort_values(["aggregator", "arm", "k", "context", "split_seed"])
        .reset_index(drop=True)
    )


def c5_headline(path: str, k: int, aggregator: str = "mean") -> pd.DataFrame:
    """Each arm at one budget, averaged over contexts and split seeds.

    `delta_accuracy` is the arm minus the full panel, which is the quantity the
    claim states a tolerance on.  Read it beside `verdict_agreement_vs_full`:
    an arm can land on the same accuracy while deciding different items, and
    only the second column notices.

    `n_seeds_virtual_better` is filled in on the `physical` row only.  It counts
    the split seeds where reconstruction beat dropping the same judges, because
    a mean difference that rests on one partition is not evidence.
    """
    df = c5_table(path)
    df = df[df["aggregator"] == aggregator]
    df = df[(df["k"] == k) | (df["arm"].isin(["full", "single_best"]))]

    out = (
        df.groupby("arm")
        .agg(
            k=("k", "first"),
            accuracy=("accuracy", "mean"),
            macro_accuracy=("macro_accuracy", "mean"),
            nll=("nll", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
            verdict_agreement=("verdict_agreement_vs_full", "mean"),
            item_rank_tau=("item_rank_kendall_tau", "mean"),
            domain_rank_tau=("domain_rank_kendall_tau", "mean"),
            n_observations=("accuracy", "size"),
        )
        .reset_index()
    )

    reference = out.loc[out["arm"] == "full", "accuracy"]
    base = float(reference.iloc[0]) if len(reference) else np.nan
    out["delta_accuracy"] = out["accuracy"] - base

    per_seed = (
        df[df["arm"].isin(["virtual", "physical"])]
        .groupby(["arm", "split_seed"])["accuracy"].mean().unstack("arm")
    )
    wins = (
        int((per_seed["virtual"] > per_seed["physical"]).sum())
        if {"virtual", "physical"} <= set(per_seed.columns) else 0
    )
    out["n_seeds_virtual_better"] = np.where(out["arm"] == "physical", wins, np.nan)
    out["n_seeds"] = df["split_seed"].nunique()

    out["label"] = out["arm"].map(lambda a: C5_PRETTY.get(a, a))
    out["rank_order"] = out["arm"].map(
        {a: i for i, a in enumerate(C5_ARM_ORDER)}
    ).fillna(len(C5_ARM_ORDER))
    return (
        out.sort_values("rank_order").drop(columns="rank_order").reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# C6 -- exchange structure
# --------------------------------------------------------------------------

C6_METHOD_ORDER = [
    "forward", "backward", "forward_trim",
    "swap2", "swap2_forward", "swap2_pq", "swap3",
]

C6_PRETTY = {
    "forward": "Forward",
    "backward": "Backward",
    "forward_trim": "Forward + trim",
    "swap2": "2-swap (backward seed)",
    "swap2_forward": "2-swap (forward seed)",
    "swap2_pq": "2-swap (priority queue)",
    "swap3": "3-swap",
}


def c6_table(path: str) -> pd.DataFrame:
    """One row per (subpanel, gamma, method) from E4, against the exact optimum."""
    payload = load(path)
    rows = []
    for index, instance in enumerate(payload["instances"]):
        for row in instance["rows"]:
            for method in payload["methods"]:
                rows.append({
                    "subpanel": index,
                    "n_judges": instance["n_judges"],
                    "gamma": row["gamma"],
                    "method": method,
                    "opt_size": row["opt_size"],
                    "size": row["sizes"][method],
                    "gap": row["gap"][method],
                    "is_exact": row["exact"][method],
                    "seconds": row["method_seconds"][method],
                    "opt_error": row["opt_error"],
                    "search_seconds": row["search_seconds"],
                })
    return (
        pd.DataFrame(rows)
        .sort_values(["method", "subpanel", "gamma"])
        .reset_index(drop=True)
    )


def c6_headline(path: str) -> pd.DataFrame:
    """Per method: how often it hit the certified optimum, and by how much it missed.

    `mean_gap` is in extra judges, so it is directly comparable with the
    section 26 planning targets, which are carried in the `meets_*` columns.
    """
    payload = load(path)
    targets = payload["summary"]["target_section_26"]

    rows = []
    for method, stats in payload["summary"]["per_method"].items():
        rows.append({
            "method": method,
            "label": C6_PRETTY.get(method, method),
            "n_instances": stats["n_instances"],
            "exact_rate": stats["exact_rate"],
            "mean_gap": stats["mean_gap"],
            "max_gap": stats["max_gap"],
            "mean_seconds": stats["mean_seconds"],
        })
    out = pd.DataFrame(rows)

    # The targets are stated for 2-swap, so they are attached to that row only.
    for name, value in targets.items():
        out[name] = np.where(out["method"] == "swap2", value, np.nan)

    out["rank_order"] = out["method"].map(
        {m: i for i, m in enumerate(C6_METHOD_ORDER)}
    ).fillna(len(C6_METHOD_ORDER))
    return (
        out.sort_values("rank_order").drop(columns="rank_order").reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# C7 -- certification reliability
# --------------------------------------------------------------------------

C7_PRETTY = {
    "empirical_fit": "Empirical, fitting split (naive)",
    "empirical_cert": "Empirical, certification split",
    "normal_uncorrected": "Normal UCB, uncorrected",
    "bernstein_uncorrected": "Bernstein UCB, uncorrected",
    "bernstein_bonferroni": "Bernstein UCB, Bonferroni (primary)",
    "bootstrap_max": "Grouped bootstrap of the maximum",
}


def _c7_rule_parts(rule: str) -> tuple:
    """Split `bernstein_bonferroni@0.05` into its rule and its delta."""
    if "@" in rule:
        base, delta = rule.split("@")
        return base, float(delta)
    return rule, np.nan


def c7_table(path: str) -> pd.DataFrame:
    """One row per (rule, delta, gamma) from E5.

    `violation_rate` is conditional on the rule having certified the case, so it
    must be read next to `certified_frac`: a rule that certifies nothing never
    violates anything and is worthless.
    """
    payload = load(path)
    rows = []
    for row in payload["summary"]["rows"]:
        base, delta = _c7_rule_parts(row["rule"])
        rows.append({
            "rule": base,
            "delta": delta,
            "gamma": row["gamma"],
            "n_cases": row["n_cases"],
            "n_certified": row["n_certified"],
            "certified_frac": row["certified_frac"],
            "n_violated": row["n_violated"],
            "violation_rate": row["violation_rate"],
            "violation_rate_hi": row["violation_rate_hi"],
        })
    out = pd.DataFrame(rows)
    out["label"] = out["rule"].map(lambda r: C7_PRETTY.get(r, r))
    return out.sort_values(["rule", "delta", "gamma"]).reset_index(drop=True)


def c7_headline(path: str, delta: float = 0.05) -> pd.DataFrame:
    """The certified rules against the empirical ones, at one delta.

    Two columns decide the claim.  `within_nominal` asks whether the violation
    rate is at or below delta, using the Wilson upper end rather than the point
    estimate so a rate of "0 out of 3" is not read as proof.  `useful` asks
    whether the rule certified anything at all.  A rule needs both.

    Rules that carry no delta -- the two empirical ones -- appear at every
    delta, because they are the comparison the claim is stated against.
    """
    df = c7_table(path)
    df = df[df["delta"].isna() | np.isclose(df["delta"], delta)]

    out = (
        df.groupby(["rule", "label"])
        .agg(
            n_cases=("n_cases", "sum"),
            n_certified=("n_certified", "sum"),
            certified_frac=("certified_frac", "mean"),
            n_violated=("n_violated", "sum"),
            worst_violation_rate=("violation_rate", "max"),
            worst_violation_rate_hi=("violation_rate_hi", "max"),
            n_gammas=("gamma", "nunique"),
        )
        .reset_index()
    )
    out["violation_rate"] = np.where(
        out["n_certified"] > 0, out["n_violated"] / out["n_certified"], np.nan
    )
    out["delta"] = delta
    out["within_nominal"] = out["worst_violation_rate_hi"] <= delta
    out["useful"] = out["n_certified"] > 0
    out["is_primary"] = out["rule"] == "bernstein_bonferroni"

    order = {r: i for i, r in enumerate(C7_PRETTY)}
    out["rank_order"] = out["rule"].map(lambda r: order.get(r, len(order)))
    return (
        out.sort_values("rank_order").drop(columns="rank_order").reset_index(drop=True)
    )


def reconstruction_table(path: str) -> pd.DataFrame:
    """Section 21.3: the same selected set, five different fitting rules."""
    payload = load(path)
    rows = []
    for k, body in payload["reconstruction_ablation_TEST"].items():
        for rule, row in body["rules"].items():
            rows.append({
                "k": int(k),
                "rule": rule,
                "kept": ",".join(body["kept"]),
                "worst_judge_tv": row["worst_judge_worst_context_tv"],
                "mean_judge_tv": row["mean_judge_worst_context_tv"],
                "verdict_agreement": row["verdict_agreement"],
                "off_simplex_frac": row["off_simplex_frac"],
                "max_weight_l1": row["max_weight_l1"],
                "min_weight": row["min_weight"],
            })
    return pd.DataFrame(rows).sort_values(["k", "rule"]).reset_index(drop=True)
