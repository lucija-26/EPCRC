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
    ax.set_xlabel("composition gap (min feasible $-$ $|S_{naive}|$)")
    ax.set_ylabel("cases")
    ax.set_title(f"mean gap {df['composition_gap'].mean():+.2f} judges", fontsize=9)
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
             sig=("n_seeds_significant", "sum"), cells=("n_seeds", "sum"))
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )
    decisive = pooled["sig"] >= 0.5 * pooled["cells"]
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
        [f"{row.label}  ({int(row.sig)}/{int(row.cells)})"
         for row in pooled.itertuples()], fontsize=8)
    ax.set_xlabel("paired difference in worst-judge TV: coverage $-$ baseline\n"
                  "(negative means coverage is better)")
    ax.set_title("C3: paired bootstrap on matched items\n"
                 "label shows cells where the whole interval excludes zero",
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
# driver
# --------------------------------------------------------------------------

def save_all(
    out_dir: str,
    e0_real: Optional[str] = None,
    e0_synth: Optional[str] = None,
    e1: Optional[str] = None,
    c3: Optional[str] = None,
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
