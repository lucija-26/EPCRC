"""Paper figures, drawn from the tables in `epcrc.report`.

Every figure here consumes a `report` DataFrame rather than raw JSON, so a plot
and the CSV beside it in the delivered zip are literally the same numbers.  The
notebooks call these to display and the export script calls them to write PNGs.

Nothing here computes a result.  If a curve looks wrong, the experiment is
wrong.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# The backend is deliberately not forced here.  Importing this module inside a
# notebook must leave inline rendering working; scripts that need a headless
# backend select it themselves before importing.
import matplotlib.pyplot as plt

from epcrc import report as R

__all__ = [
    "fig_c1_real",
    "fig_c1_synthetic",
    "fig_c2_frontier",
    "fig_c3_headline",
    "fig_c3_paired",
    "fig_c3_split_robustness",
    "fig_c3_cost",
    "fig_c3_reconstruction",
    "fig_c4_stress",
    "fig_c5_downstream",
    "fig_c6_exchange",
    "fig_c7_certification",
    "fig_c8_sparse",
    "fig_backbone",
    "fig_e7_transfer",
    "save_all",
]

# One colour for the method under test, one for everything it has to beat.
COVERAGE_COLOUR = "#1f4e79"
BASELINE_COLOUR = "#b0b0b0"
FLOOR_COLOUR = "#c0504d"
ACCENT = "#e8a33d"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------

def fig_c1_real(e0_real_path: str) -> plt.Figure:
    """The C1 picture on real judges: certified-removable, then removed together.

    Each point is one (split seed, tolerance) at which at least two judges
    passed their own leave-one-out test.  The x axis is the tolerance they all
    passed; the y axis is what the panel actually achieves once they are all
    deleted.  Anything above the diagonal is a certificate that did not
    compose.
    """
    df = R.c1_table(e0_real_path)
    df = df[df["gamma_source"] == "loo_breakpoint"]
    df = df[df["n_individually_removable"] >= 2].copy()
    finite = df[np.isfinite(df["naive_coverage"])]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ax = axes[0]
    lim = [0, float(max(finite["naive_coverage"].max(), finite["gamma"].max())) * 1.08]
    ax.plot(lim, lim, color="black", lw=1, ls="--", label="certificate would compose")
    ax.fill_between(lim, lim, lim[1], color=FLOOR_COLOUR, alpha=0.08)
    ax.scatter(finite["gamma"], finite["naive_coverage"],
               s=26, c=COVERAGE_COLOUR, alpha=0.8, edgecolor="white", lw=0.5,
               label="joint deletion of $R(\\gamma)$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("tolerance $\\gamma$ every removed judge passed")
    ax.set_ylabel("error after removing them together")
    n_viol = int(df["naive_coverage"].gt(df["gamma"]).sum())
    n_empty = len(df) - len(finite)
    title = f"violations: {n_viol} / {len(df)} cases"
    if n_empty:
        title += f"\n({n_empty} off scale: $R(\\gamma)$ emptied the panel)"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    gaps = df["composition_gap"].value_counts().sort_index()
    ax.bar(gaps.index, gaps.values, color=COVERAGE_COLOUR, width=0.6)
    ax.axvline(0, color="black", lw=1, ls="--")
    # Past E0's exhaustive budget the minimum feasible panel is only bounded
    # above, so the gap is bounded above too.  Drawing that as a bare number
    # would present an upper bound as the measured gap, so the axis says which
    # it is.  The left panel is unaffected: it is a single coverage evaluation.
    bounded = not bool(df["min_feasible_is_exact"].all())
    relation = "$\\leq$ " if bounded else ""
    ax.set_xlabel(f"composition gap ({relation}min feasible $-$ $|S_{{naive}}|$)")
    ax.set_ylabel("cases")
    title = f"mean gap {relation}{df['composition_gap'].mean():+.2f} judges"
    if bounded:
        n_bounded = int((~df["min_feasible_is_exact"]).sum())
        title += f"\n({n_bounded} / {len(df)} bounded, not enumerated)"
    ax.set_title(title, fontsize=9)
    ax.set_xticks(sorted(gaps.index))

    fig.suptitle("C1: individual redundancy certificates do not compose (real panel)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_c1_synthetic(e0_path: str) -> plt.Figure:
    """The same claim on the two controlled constructions, where truth is known."""
    df = R.c1_headline(e0_path, gamma_source=None)
    instances = sorted(df["instance"].unique())

    fig, axes = plt.subplots(1, len(instances), figsize=(4.2 * len(instances), 3.2),
                             squeeze=False)
    for ax, name in zip(axes[0], instances):
        sub = df[df["instance"] == name].sort_values("gamma")
        ax.plot(sub["gamma"], sub["naive_size"], "o-", color=BASELINE_COLOUR,
                label="$|S_{naive}|$ (one-at-a-time audit)")
        ax.plot(sub["gamma"], sub["min_feasible"], "s-", color=COVERAGE_COLOUR,
                label="minimum jointly feasible")
        ax.fill_between(sub["gamma"], sub["naive_size"], sub["min_feasible"],
                        color=FLOOR_COLOUR, alpha=0.15, label="composition gap")
        ax.set_xscale("log")
        ax.set_xlabel("tolerance $\\gamma$")
        ax.set_ylabel("panel size")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("C1: controlled constructions", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C2
# --------------------------------------------------------------------------

def fig_c2_frontier(e1_path: str, c3_path: Optional[str] = None) -> plt.Figure:
    """The compression frontier: held-out worst-judge error against budget.

    The low-rank floor is drawn when C3 results are available, because it is
    the bound no subset selection can beat -- it is allowed to use a basis that
    is not any real judge, so it is not deployable, only a reference.
    """
    df = R.c2_table(e1_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax = axes[0]
    rnd = df[df["method"] == "random"]
    if len(rnd):
        band = rnd.groupby("k")["worst_judge_tv"]
        ax.fill_between(band.mean().index, band.min(), band.max(),
                        color=BASELINE_COLOUR, alpha=0.35, label="random draws (range)")

    for method, style in (("coverage_backward", "o-"), ("coverage_forward", "^--"),
                          ("exhaustive", "*-")):
        sub = df[df["method"] == method].sort_values("k")
        if not len(sub):
            continue
        ax.plot(sub["k"], sub["worst_judge_tv"], style,
                color=COVERAGE_COLOUR if method != "exhaustive" else ACCENT,
                label=R.PRETTY[method], ms=5)

    for method, colour in (("top_accuracy", "#7f7f7f"), ("one_per_family", "#555555")):
        sub = df[df["method"] == method].sort_values("k")
        if len(sub):
            ax.plot(sub["k"], sub["worst_judge_tv"], ":", color=colour,
                    label=R.PRETTY[method], lw=1.2)

    if c3_path and os.path.exists(c3_path):
        floor = R.lowrank_floor_table(c3_path)
        pca = floor[floor["method"] == "pca"].sort_values("k")
        ax.plot(pca["k"], pca["worst_judge_tv"], "-", color=FLOOR_COLOUR, lw=1.2,
                label="rank-$k$ floor (not deployable)")

    ax.set_xlabel("physical panel size $k$")
    ax.set_ylabel("worst-judge worst-context mean TV")
    ax.set_title("held-out reconstruction error (TEST)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    sub = df[df["method"] == "coverage_backward"].sort_values("k")
    ax.plot(sub["k"], sub["verdict_agreement"], "o-", color=COVERAGE_COLOUR,
            label="verdict agreement")
    ax.plot(sub["k"], sub["p95_item_tv"], "s--", color=FLOOR_COLOUR,
            label="95th percentile item TV")
    ax.plot(sub["k"], sub["median_item_tv"], "^:", color=ACCENT,
            label="median item TV")
    ax.set_xlabel("physical panel size $k$")
    ax.set_ylabel("fraction / TV")
    ax.set_title("what a downstream user sees", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("C2: the physical-to-virtual compression frontier", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------

def fig_c3_headline(c3_path: str) -> plt.Figure:
    """Head-to-head against every baseline, with bootstrap intervals."""
    df = R.c3_headline(c3_path).iloc[::-1].reset_index(drop=True)
    colours = [COVERAGE_COLOUR if c else BASELINE_COLOUR for c in df["is_coverage"]]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(df) + 1.4))
    ax.barh(y, df["worst_judge_tv"], color=colours, height=0.68)
    ax.errorbar(df["worst_judge_tv"], y,
                xerr=[df["worst_judge_tv"] - df["boot_lo"],
                      df["boot_hi"] - df["worst_judge_tv"]],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("worst-judge worst-context mean TV (lower is better)")
    ks = df["k_values"].iloc[0]
    ax.set_title(f"C3: averaged over $k \\in$ {ks}, 5 split seeds\n"
                 f"bars are item bootstrap intervals", fontsize=9)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


def fig_c3_paired(c3_path: str) -> plt.Figure:
    """The paired difference against each baseline on the *same* resampled items.

    This is the test that can actually decide the comparison.  Two independently
    computed intervals share most of their sampling noise, because both methods
    are scored on one set of items, so asking whether those intervals overlap
    throws the shared noise away.  Differencing on matched resamples keeps it.
    """
    df = R.c3_paired_table(c3_path)
    pooled = (
        df.groupby(["method", "label"], as_index=False)
        .agg(delta=("delta", "mean"), lo=("lo", "mean"), hi=("hi", "mean"),
             won=("n_seeds_reference_better", "sum"),
             lost=("n_seeds_baseline_better", "sum"),
             cells=("n_seeds", "sum"))
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )
    # Decided cells that went against the reference are losses, so counting
    # them towards a majority would colour a bar as a win it did not earn.
    decisive = pooled["won"] >= 0.5 * pooled["cells"]
    y = np.arange(len(pooled))

    fig, ax = plt.subplots(figsize=(6.6, 0.36 * len(pooled) + 1.6))
    ax.barh(y, pooled["delta"],
            color=[COVERAGE_COLOUR if d else BASELINE_COLOUR for d in decisive],
            height=0.66)
    ax.errorbar(pooled["delta"], y,
                xerr=[pooled["delta"] - pooled["lo"], pooled["hi"] - pooled["delta"]],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.label}  ({int(row.won)}W {int(row.lost)}L / {int(row.cells)})"
         for row in pooled.itertuples()], fontsize=8)
    ax.set_xlabel("paired difference in worst-judge TV: coverage $-$ baseline\n"
                  "(negative means coverage is better)")
    ax.set_title("C3: paired bootstrap on matched items\n"
                 "label shows decided cells won and lost by coverage",
                 fontsize=9)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


def fig_c3_split_robustness(c3_path: str) -> plt.Figure:
    """Across-seed spread: the headline must not rest on one favourable split."""
    rob = R.c3_split_robustness(c3_path).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rob))
    is_cov = rob["method"].isin(R.COVERAGE_METHODS)

    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(rob) + 1.5))
    ax.hlines(y, rob["min"], rob["max"],
              color=[COVERAGE_COLOUR if c else BASELINE_COLOUR for c in is_cov],
              lw=4, alpha=0.55)
    ax.scatter(rob["mean"], y, s=30, zorder=3,
               c=[COVERAGE_COLOUR if c else "#4d4d4d" for c in is_cov],
               edgecolor="white", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(rob["label"], fontsize=8)
    ax.set_xlabel("worst-judge worst-context mean TV")
    ax.set_title(f"C3: min-to-max across {int(rob['n_seeds'].iloc[0])} split seeds\n"
                 f"dot is the across-seed mean", fontsize=9)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


def fig_c3_cost(c3_path: str) -> plt.Figure:
    """Accuracy against measured inference cost, not against judge count.

    Judges are not equally expensive, so equal-$k$ is not equal-cost; this is
    the panel the plan asks for.
    """
    cost = R.c3_cost_table(c3_path)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(cost["panel_seconds"], cost["worst_judge_tv"], "o-",
            color=COVERAGE_COLOUR)
    for _, row in cost.iterrows():
        ax.annotate(f"k={int(row['k'])}",
                    (row["panel_seconds"], row["worst_judge_tv"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    total = cost["panel_seconds"].max()
    ax.axvline(total, color=BASELINE_COLOUR, ls="--", lw=1)
    ax.text(total, ax.get_ylim()[1], " full panel", fontsize=7, va="top",
            color="#666666")
    ax.set_xlabel("measured panel inference time (seconds)")
    ax.set_ylabel("worst-judge worst-context mean TV")
    ax.set_title("C3: error against measured cost", fontsize=9)
    fig.tight_layout()
    return fig


def fig_c3_reconstruction(c3_path: str) -> plt.Figure:
    """Same selected judges, five fitting rules.

    The unconstrained rules can win on TV while leaving the simplex, which
    means their output is not a probability distribution and cannot be shipped
    as a judge verdict; `off_simplex_frac` is the price they pay.
    """
    df = R.reconstruction_table(c3_path)
    rules = ["simplex", "nonneg", "ridge", "nearest", "uniform"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for rule in rules:
        sub = df[df["rule"] == rule].sort_values("k")
        style = "o-" if rule == "simplex" else "--"
        colour = COVERAGE_COLOUR if rule == "simplex" else None
        axes[0].plot(sub["k"], sub["worst_judge_tv"], style, label=rule,
                     color=colour, lw=1.6 if rule == "simplex" else 1.1)
        axes[1].plot(sub["k"], sub["off_simplex_frac"], style, label=rule,
                     color=colour, lw=1.6 if rule == "simplex" else 1.1)

    axes[0].set_xlabel("physical panel size $k$")
    axes[0].set_ylabel("worst-judge worst-context mean TV")
    axes[0].set_title("reconstruction error", fontsize=9)
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("physical panel size $k$")
    axes[1].set_ylabel("fraction of items off the simplex")
    axes[1].set_title("output is not a distribution", fontsize=9)
    axes[1].legend(fontsize=7)

    fig.suptitle("C3: the reconstruction rule matters as much as the subset",
                 fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C4
# --------------------------------------------------------------------------

def fig_c4_stress(c4_path: str) -> plt.Figure:
    """What selecting on clean items alone costs, and who pays for it.

    Left: the advantage of robust selection at each budget, as baseline minus
    robust, with the full across-seed range drawn rather than a standard error.
    A mean above zero with a range straddling it is a split result, and the band
    is what makes that visible instead of hiding it in an error bar.

    Right: how many stress specialists each arm keeps, which is the mechanism
    the left panel is supposed to be explaining.
    """
    head = R.c4_headline(c4_path)
    rows = R.c4_table(c4_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax = axes[0]
    colours = {"clean_select": ACCENT, "clean_pipeline": FLOOR_COLOUR}
    for baseline, group in head.groupby("baseline"):
        group = group.sort_values("k")
        colour = colours.get(baseline, BASELINE_COLOUR)
        ax.plot(group["k"], group["delta_mean"], "o-", color=colour,
                label=f"vs {baseline}")
        ax.fill_between(group["k"], group["delta_min"], group["delta_max"],
                        color=colour, alpha=0.15)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("physical panel size $k$")
    ax.set_ylabel("baseline $-$ robust worst-context TV")
    ax.set_title("above zero: robust selection wins\n(band = across-seed range)",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    kept = (
        rows.groupby(["arm", "k"])["n_specialists_kept"].mean().reset_index()
    )
    styles = {"robust": (COVERAGE_COLOUR, "s-"),
              "clean_select": (ACCENT, "o-"),
              "clean_pipeline": (FLOOR_COLOUR, "^-")}
    for arm, group in kept.groupby("arm"):
        group = group.sort_values("k")
        colour, marker = styles.get(arm, (BASELINE_COLOUR, "o-"))
        ax.plot(group["k"], group["n_specialists_kept"], marker, color=colour,
                label=arm)
    total = int(rows["n_specialists_total"].max())
    ax.axhline(total, color="black", lw=1, ls=":",
               label=f"all {total} specialists")
    ax.set_xlabel("physical panel size $k$")
    ax.set_ylabel("stress specialists retained")
    ax.set_title("the mechanism: who gets kept", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("C4: selecting on clean items alone retires stress specialists",
                 fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# C5 -- downstream preservation
# --------------------------------------------------------------------------

def fig_c5_downstream(c5_path: str, aggregator: str = "mean") -> plt.Figure:
    """What reconstruction buys downstream, budget by budget.

    Left: how far each arm's accuracy sits from the full panel's.  Right: how
    often each arm decides the same item the same way.  The two are separate
    panels because an arm can match the accuracy while deciding different items,
    and only the right panel notices that.

    `physical` is the comparison that matters: same budget, same judges, no
    reconstruction.  The shaded gap between the two curves is the claim.
    """
    df = R.c5_table(c5_path)
    df = df[df["aggregator"] == aggregator]

    full = df[df["arm"] == "full"]["accuracy"].mean()
    budgets = sorted(df[df["arm"] == "virtual"]["k"].unique())

    def curve(arm: str, column: str):
        sub = df[(df["arm"] == arm) & df["k"].isin(budgets)]
        grouped = sub.groupby("k")[column].mean()
        return [grouped.get(k, np.nan) for k in budgets]

    arms = ["virtual", "physical", "top_accuracy", "one_per_family", "random"]
    styles = {
        "virtual": dict(color=COVERAGE_COLOUR, lw=2.0, marker="o", zorder=3),
        "physical": dict(color=FLOOR_COLOUR, lw=2.0, marker="s", zorder=3),
        "top_accuracy": dict(color=ACCENT, lw=1.2, marker="^", alpha=0.9),
        "one_per_family": dict(color=BASELINE_COLOUR, lw=1.0, marker="v"),
        "random": dict(color=BASELINE_COLOUR, lw=1.0, ls=":", marker="d"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    ax = axes[0]
    virtual = np.array(curve("virtual", "accuracy")) - full
    physical = np.array(curve("physical", "accuracy")) - full
    ax.fill_between(budgets, virtual, physical, color=COVERAGE_COLOUR, alpha=0.10)
    for arm in arms:
        ax.plot(budgets, np.array(curve(arm, "accuracy")) - full,
                label=R.C5_PRETTY.get(arm, arm), **styles[arm])
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("physical judges executed, $k$")
    ax.set_ylabel("accuracy $-$ full panel accuracy")
    ax.set_title(f"decision quality ({aggregator} aggregate)", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    for arm in arms:
        ax.plot(budgets, curve(arm, "verdict_agreement_vs_full"), **styles[arm])
    ax.set_xlabel("physical judges executed, $k$")
    ax.set_ylabel("items decided as the full panel decided them")
    ax.set_title("agreement with the full panel", fontsize=9)

    fig.suptitle("C5: reconstruction preserves the panel's decisions better than "
                 "dropping the same judges", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C6 -- exchange structure
# --------------------------------------------------------------------------

def fig_c6_exchange(c6_path: str) -> plt.Figure:
    """How close each method gets to the certified optimum, and what it costs.

    Left: the share of instances solved exactly, with the section 26 target
    drawn as a line.  Right: mean extra judges against mean runtime, which is
    where the 3-swap question is settled -- it can only earn its cost by sitting
    lower *and* not far to the right.
    """
    head = R.c6_headline(c6_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    ax = axes[0]
    colours = [
        COVERAGE_COLOUR if m.startswith("swap") else BASELINE_COLOUR
        for m in head["method"]
    ]
    ax.bar(range(len(head)), head["exact_rate"], color=colours, width=0.62)
    ax.axhline(0.70, color=FLOOR_COLOUR, lw=1.2, ls="--",
               label="section 26 target (0.70)")
    ax.set_xticks(range(len(head)))
    ax.set_xticklabels(head["label"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("instances solved exactly")
    ax.set_ylim(0, 1.05)
    ax.set_title("exact-optimum rate", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    for _, row in head.iterrows():
        colour = COVERAGE_COLOUR if row["method"].startswith("swap") else BASELINE_COLOUR
        ax.scatter(row["mean_seconds"], row["mean_gap"], s=60, color=colour,
                   edgecolor="white", lw=0.6, zorder=3)
        ax.annotate(row["label"], (row["mean_seconds"], row["mean_gap"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.5)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("mean seconds per instance (log scale)")
    ax.set_ylabel("mean extra judges vs optimum")
    ax.set_title("does the extra search pay for itself?", fontsize=9)

    fig.suptitle("C6: low-order exchange closes most of the gap to the certified "
                 "optimum", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C7 -- certification reliability
# --------------------------------------------------------------------------

def fig_c7_certification(c7_path: str, delta: float = 0.05) -> plt.Figure:
    """Violation rate against the nominal level, and what certifying costs.

    Left: for each rule and tolerance, how often a certified panel broke its
    promise on the locked split.  Anything above the dashed line is a rule whose
    stated confidence is not being kept.

    Right: the price of that confidence.  A rule that certifies nothing sits at
    zero on the left and is worthless, so the two panels must be read together.
    """
    df = R.c7_table(c7_path)
    df = df[df["delta"].isna() | np.isclose(df["delta"], delta)]
    rules = [r for r in R.C7_PRETTY if r in set(df["rule"])]

    def style(rule: str) -> dict:
        if rule == "bernstein_bonferroni":
            return dict(color=COVERAGE_COLOUR, lw=2.2, marker="o", zorder=3)
        if rule.startswith("empirical"):
            return dict(color=FLOOR_COLOUR, lw=1.6, marker="s")
        if rule == "bootstrap_max":
            return dict(color=ACCENT, lw=1.6, marker="^")
        return dict(color=BASELINE_COLOUR, lw=1.1, marker="d", ls=":")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    ax = axes[0]
    for rule in rules:
        sub = df[df["rule"] == rule].sort_values("gamma")
        ax.plot(sub["gamma"], sub["violation_rate"],
                label=R.C7_PRETTY[rule], **style(rule))
    ax.axhline(delta, color="black", lw=1, ls="--",
               label=f"nominal level ({delta:g})")
    ax.set_xlabel("stated tolerance $\\gamma$")
    ax.set_ylabel("certified panels that broke $\\gamma$ on TEST")
    ax.set_title("is the promise kept?", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")

    ax = axes[1]
    for rule in rules:
        sub = df[df["rule"] == rule].sort_values("gamma")
        ax.plot(sub["gamma"], sub["certified_frac"], **style(rule))
    ax.set_xlabel("stated tolerance $\\gamma$")
    ax.set_ylabel("cases the rule was willing to certify")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("what the confidence costs", fontsize=9)

    fig.suptitle(f"C7: an independent split and a simultaneous bound, at "
                 f"$\\delta$ = {delta:g}", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C8 -- how sparse a certificate can be
# --------------------------------------------------------------------------

def fig_c8_sparse(e6_path: str) -> plt.Figure:
    """What capping the support costs, and whether the cap binds at all.

    Left: error as a multiple of the uncapped fit.  A cap that costs nothing is
    a flat line at one.

    Right: how many judges the uncapped fit actually used.  This is the panel
    that decides how to read the left one -- a cap set above the support the
    solver would have chosen anyway is not a constraint, so the two must be
    read together or the left panel looks like a stronger result than it is.
    """
    df = R.c8_sparse_table(e6_path)
    caps = sorted(c for c in set(df["support_cap"]) if c != "none")
    budgets = sorted(set(df["k"]))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    ax = axes[0]
    for i, k in enumerate(budgets):
        sub = df[(df["k"] == k) & (df["support_cap"] != "none")]
        sub = sub.set_index("support_cap").loc[caps]
        ax.plot(caps, sub["ratio_to_uncapped"], marker="o",
                color=plt.cm.viridis(i / max(len(budgets) - 1, 1)),
                label=f"k = {k}")
    ax.axhline(1.0, color="black", lw=1, ls="--", label="uncapped")
    ax.set_xticks(caps)
    ax.set_xlabel("judges allowed per virtual judge $r$")
    ax.set_ylabel("worst-judge TV / uncapped")
    ax.set_title("what the cap costs", fontsize=9)
    ax.legend(fontsize=6.5)

    ax = axes[1]
    uncapped = df[df["support_cap"] == "none"].sort_values("k")
    ax.bar([str(k) for k in uncapped["k"]], uncapped["mean_support_size"],
           color=COVERAGE_COLOUR, width=0.55)
    for cap in caps:
        ax.axhline(cap, color=BASELINE_COLOUR, lw=0.8, ls=":")
    ax.set_xlabel("retained judges $k$")
    ax.set_ylabel("mean judges used, no cap")
    ax.set_title("does the cap bind?", fontsize=9)

    fig.suptitle("C8: three or four judges per virtual judge is what the "
                 "uncapped fit already chooses", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# backbone -- which judges every optimum must contain
# --------------------------------------------------------------------------

def fig_backbone(backbone_path: str) -> plt.Figure:
    """The three section-20 categories against tolerance.

    A judge counts as mandatory only if it is mandatory under every split seed,
    so `unstable` is its own band rather than being folded into one of the
    three.  A band that grows with tolerance is redundancy appearing.
    """
    head = R.backbone_headline(backbone_path)
    head = head[head["n_seeds_feasible"] > 0].sort_values("gamma")
    gammas = list(head["gamma"])

    # Counted off the per-judge verdicts rather than off the headline strings,
    # because the headline names only the two extreme categories.  Deriving the
    # optional band by subtraction would hide the judges that fall in some
    # optimum but not all -- which is the section 20 category the whole
    # question is about.
    per_judge = R.backbone_per_judge(backbone_path)
    counts = (per_judge.groupby(["gamma", "verdict"]).size()
              .unstack(fill_value=0).reindex(gammas, fill_value=0))
    bands = [counts.get(v, 0 * counts.iloc[:, 0])
             for v in ("mandatory", "optional", "unstable", "nonessential")]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    ax = axes[0]
    ax.stackplot(
        gammas, *bands,
        labels=["in every optimum", "in some optima",
                "category varies by seed", "in no optimum"],
        colors=[COVERAGE_COLOUR, "#7fa8cc", ACCENT, BASELINE_COLOUR], alpha=0.9)
    ax.set_xlabel("stated tolerance $\\gamma$")
    ax.set_ylabel("judges")
    ax.set_title("how the panel splits", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axes[1]
    ax.plot(gammas, head["k_star_min"], marker="o", color=COVERAGE_COLOUR,
            label="smallest certified panel $k^*$")
    ax.fill_between(gammas, head["k_star_min"], head["k_star_max"],
                    color=COVERAGE_COLOUR, alpha=0.2,
                    label="range over split seeds")
    ax.set_xlabel("stated tolerance $\\gamma$")
    ax.set_ylabel("judges in the smallest feasible panel")
    ax.set_title("and how far it can shrink", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")

    fig.suptitle("Backbone: nothing is redundant until the tolerance is "
                 "loosened well past the target", fontsize=10)
    fig.tight_layout()
    return fig


def fig_e7_transfer(e7_path: str) -> plt.Figure:
    """E7 in two panels: where the basis lands off-benchmark, and what refitting buys.

    Left is the ordering question -- the coverage basis against its baselines on
    a benchmark that had no part in choosing it, with the oracle drawn as a
    floor so the reader can see it is a ceiling on reselection and not a rival.
    Right is the plan's calibration-sample efficiency, on a log x axis because
    the sizes are spaced by multiples.
    """
    table = R.e7_table(e7_path)
    calib = R.e7_calibration(e7_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax = axes[0]
    reference = R.PRETTY["coverage_backward"]
    grouped = table.groupby(["method", "k"], as_index=False)["frozen_tv"].mean()
    for method, block in grouped.groupby("method"):
        block = block.sort_values("k")
        is_ours = method == reference
        ax.plot(block["k"], block["frozen_tv"],
                marker="o" if is_ours else "s",
                ms=4.5 if is_ours else 3.5,
                lw=2.0 if is_ours else 1.0,
                color=COVERAGE_COLOUR if is_ours else BASELINE_COLOUR,
                zorder=3 if is_ours else 2,
                label=method if is_ours else None)
    oracle = table.groupby("k", as_index=False)["oracle_tv"].mean().sort_values("k")
    ax.plot(oracle["k"], oracle["oracle_tv"], lw=1.2, ls="--", color=FLOOR_COLOUR,
            label="oracle reselection (ceiling)")
    ax.plot([], [], marker="s", ms=3.5, lw=1.0, color=BASELINE_COLOUR,
            label="baselines")
    ax.set_xlabel("physical judges kept, k")
    ax.set_ylabel("worst-judge worst-context TV")
    ax.set_title("Transfer: frozen basis and weights")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for k, block in calib.groupby("k"):
        block = block.sort_values("calibration_pairs")
        ax.plot(block["calibration_pairs"], block["tv"], marker="o", ms=3.5,
                lw=1.2, label=f"k = {int(k)}")
    ax.set_xscale("log")
    ax.set_xlabel("calibration pairs from the new benchmark")
    ax.set_ylabel("worst-judge worst-context TV")
    ax.set_title("Refitting the weights only")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.tight_layout()
    return fig


def save_all(
    out_dir: str,
    e0_real: Optional[str] = None,
    e0_synth: Optional[str] = None,
    e1: Optional[str] = None,
    c3: Optional[str] = None,
    c4: Optional[str] = None,
    c5: Optional[str] = None,
    c6: Optional[str] = None,
    c7: Optional[str] = None,
    e6: Optional[str] = None,
    backbone: Optional[str] = None,
    e7: Optional[str] = None,
) -> list:
    """Write every figure whose inputs exist; return the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("c1_real_composition.png", fig_c1_real, (e0_real,)),
        ("c1_synthetic_gap.png", fig_c1_synthetic, (e0_synth,)),
        ("c2_frontier.png", fig_c2_frontier, (e1, c3)),
        ("c3_headline.png", fig_c3_headline, (c3,)),
        ("c3_paired.png", fig_c3_paired, (c3,)),
        ("c3_split_robustness.png", fig_c3_split_robustness, (c3,)),
        ("c3_cost.png", fig_c3_cost, (c3,)),
        ("c3_reconstruction.png", fig_c3_reconstruction, (c3,)),
        ("c4_stress.png", fig_c4_stress, (c4,)),
        ("c5_downstream.png", fig_c5_downstream, (c5,)),
        ("c5_e7_transfer.png", fig_e7_transfer, (e7,)),
        ("c6_exchange.png", fig_c6_exchange, (c6,)),
        ("c7_certification.png", fig_c7_certification, (c7,)),
        ("c8_sparse.png", fig_c8_sparse, (e6,)),
        ("backbone.png", fig_backbone, (backbone,)),
    ]

    written = []
    for name, func, args in jobs:
        if args[0] is None or not os.path.exists(args[0]):
            continue
        fig = func(*args)
        path = os.path.join(out_dir, name)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written
