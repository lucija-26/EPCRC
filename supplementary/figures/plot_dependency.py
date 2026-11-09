#!/usr/bin/env python3
"""Render paper figures from frozen core20 JSONs; no fitting or resampling.

Run from any directory with the project CPU environment. All source files are
read-only. PNG/PDF/SVG, exact plotted-data CSVs and SHA-256 provenance are saved
under the core20 results directory by default.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "results/supplementary/real"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#555555"
PUBLISHED_CERT = {20260818: .216, 20260819: .211, 20260820: .209,
                  20260821: .214, 20260822: .208}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output: Path, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax, gamma: float) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.9", lw=.7)
    ax.axvline(gamma, color=GRAY, lw=1.1, ls=(0, (4, 3)), zorder=1)


def dot_interval(ax, x: float, y: float, interval, color, marker="o") -> None:
    ax.errorbar(x, y, xerr=[[x-interval[0]], [interval[1]-x]],
                fmt=marker, color=color, ecolor=color, markersize=6,
                elinewidth=1.6, capsize=4, capthick=1.1, zorder=3)


def main_figure(main: dict, output: Path) -> None:
    uncertainty = main["uncertainty"]
    overall = uncertainty["overall"]
    gamma = main["gamma"]
    ids = main["removed_judge_ids"]
    rows = []
    scopes = [("panel", overall)] + [(r["judge_id"], r)
                                      for r in uncertainty["per_removed_judge"]]
    for scope, item in scopes:
        for statistic, key, interval in [
            ("individual", "individual_max_tv" if scope == "panel" else
             "individual_worst_context_tv", "individual_simultaneous_interval"),
            ("joint", "joint_max_tv" if scope == "panel" else
             "joint_worst_context_tv", "joint_simultaneous_interval"),
            ("joint_minus_individual", "joint_minus_individual", "gap_simultaneous_interval"),
        ]:
            rows.append(dict(scope=scope, split="TEST", statistic=statistic,
                             estimate=item[key], lower=item[interval][0], upper=item[interval][1],
                             alpha=uncertainty["alpha"], gamma=gamma,
                             bootstrap_repetitions=uncertainty["repetitions"],
                             n_groups=uncertainty["n_groups"],
                             interval="conditional simultaneous bootstrap band"))
    write_csv(output / "main_test_risks.csv", rows)
    A = np.asarray(main["dependency"]["A_removed_to_removed"])
    write_csv(output / "dependency_weights.csv", [
        dict(target=target, donor=donor, weight=float(A[i, j]))
        for i, target in enumerate(ids) for j, donor in enumerate(ids)])

    fig, (ax, hm) = plt.subplots(1, 2, figsize=(8.9, 4.6),
                                gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.14, right=.94, bottom=.32, top=.84, wspace=.50)
    ax.set_title("(a) Primary TEST risk", loc="left", pad=15, weight="bold")
    clean_axis(ax, gamma)
    for y, kind, color in [(1, "individual", BLUE), (0, "joint", ORANGE)]:
        value = overall[f"{kind}_max_tv"]
        dot_interval(ax, value, y, overall[f"{kind}_simultaneous_interval"], color)
        ax.annotate(f"{value:.4f}", (value, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_yticks([1, 0], ["Individual LOO\nmaximum", "Joint refit\nmaximum"])
    ax.set_ylim(-.58, 1.58)
    ax.set_xlim(.11, .235)
    ax.set_xticks([.12, .16, .20])
    ax.set_xlabel("Worst endpoint/context mean TV")
    ax.text(gamma+.002, .96, r"$\gamma=0.20$", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9, color=GRAY)
    gap = overall["joint_minus_individual"]
    lo, hi = overall["gap_simultaneous_interval"]

    hm.set_title("(b) LOO dependency weights", loc="left", pad=15, weight="bold")
    edges = np.arange(len(ids)+1)-.5
    im = hm.pcolormesh(edges, edges, A, cmap="Blues", vmin=0, vmax=.5,
                       shading="flat", rasterized=False)
    hm.set_aspect("equal")
    hm.set_xlim(-.5, len(ids)-.5)
    hm.set_ylim(len(ids)-.5, -.5)
    for (i, j), value in np.ndenumerate(A):
        hm.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10,
                color="white" if value > .28 else "#172538")
    hm.set_xticks(range(len(ids)), ids)
    hm.set_yticks(range(len(ids)), ids)
    hm.set_xlabel("Donor endpoint")
    hm.set_ylabel("Target endpoint")
    for spine in hm.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=hm, fraction=.045, pad=.06, ticks=[0, .25, .5])
    cb.solids.set_rasterized(False)
    cb.set_label("Weight", fontsize=9)
    cb.outline.set_visible(False)
    hm.text(.5, -.27, r"$A=W_{RR}$; " + rf"$\rho(A)={main['dependency']['spectral_radius']:.3f}$",
            transform=hm.transAxes, ha="center", va="top", fontsize=10)
    fig.text(.02, .035,
             f"Joint − individual = {gap:.4f}; 95% simultaneous interval [{lo:.4f}, {hi:.4f}].\n"
             "Joint risk interval crosses γ: threshold exceedance is not established.\n"
             f"95% conditional bands; frozen FIT/CERT; {uncertainty['repetitions']:,} paired bootstraps "
             f"of {uncertainty['n_groups']} source-item groups.",
             fontsize=8.5, va="bottom", color="#333333")
    save(fig, output, "primary_test_and_dependencies")


def composition_figure(main: dict, output: Path) -> None:
    comp = main["dependency"]["split_comparison"]["TEST"]
    direct = np.asarray(comp["direct_refit_context_tv"]).max(axis=0)
    composed = np.asarray(comp["composed_context_tv"]).max(axis=0)
    tight = np.asarray(comp["bound_context_tv_unclipped"]).max(axis=0)
    robust = np.asarray(comp["theorem2_robust_residual_bound_per_judge"])
    ids = main["removed_judge_ids"]
    write_csv(output / "composition_test.csv", [
        dict(judge_id=j, split="TEST", direct_refit_tv=direct[i],
             composed_tv=composed[i], contextwise_bound=tight[i], theorem2_robust_bound=robust[i],
             gamma=main["gamma"], spectral_radius=main["dependency"]["spectral_radius"],
             interpretation="Observed-split deterministic sufficient bound for composed witnesses")
        for i, j in enumerate(ids)])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.subplots_adjust(left=.12, right=.96, bottom=.24, top=.74)
    clean_axis(ax, main["gamma"])
    y = np.arange(len(ids))[::-1]
    for values, dy, color, marker, label in [
        (direct, .14, BLUE, "o", "Direct joint refit"),
        (composed, 0, GREEN, "s", "Composed LOO witnesses"),
        (robust, -.14, ORANGE, "D", "Theorem 2 bound"),
    ]:
        ax.scatter(values, y+dy, color=color, marker=marker, s=43, label=label, zorder=3)
    ax.set_yticks(y, ids)
    ax.set_ylim(-.45, len(ids)-.55)
    ax.set_xlim(.12, .435)
    ax.set_xlabel("Worst-context mean TV / deterministic upper bound")
    ax.text(.202, .96, r"$\gamma=0.20$", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color=GRAY)
    fig.suptitle("Primary TEST: refitting improves on mechanical composition",
                 x=.12, y=.97, ha="left", fontsize=12, weight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.54, .90),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.6)
    fig.text(.02, .035,
             "The bound concerns the composed witnesses on this observed split; it is sufficient, not necessary.\n"
             "A bound above γ neither proves that all joint replacements fail nor certifies population risk.",
             fontsize=8.5, va="bottom", color="#333333")
    save(fig, output, "composition_and_bound")


def five_split_figure(fixed: dict, output: Path) -> None:
    reports = fixed["split_reports"]
    repetitions = reports[0]["splits"]["TEST"]["bootstrap"]["repetitions"]
    rows = []
    for report in reports:
        seed = report["metadata"]["split_manifest_seed"]
        for role in ("CERT", "TEST"):
            stat = report["splits"][role]
            boot = stat["bootstrap"]
            overall = boot["overall"]
            for method in ("individual", "joint"):
                interval = overall[f"{method}_simultaneous_interval"]
                rows.append(dict(split_seed=seed, analysis_role=report["analysis_role"],
                                 data_role=role, method=method, estimate=overall[f"{method}_max_tv"],
                                 lower=interval[0], upper=interval[1],
                                 gamma=fixed["gamma"], alpha=boot["alpha"],
                                 bootstrap_repetitions=boot["repetitions"], n_groups=boot["n_groups"],
                                 paper_joint_cert_point=(PUBLISHED_CERT[seed]
                                                         if role == "CERT" and method == "joint" else ""),
                                 manifest_sha256=report["metadata"]["split_manifest_sha256"]))
    write_csv(output / "fixed_five_splits.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.15, right=.97, bottom=.24, top=.78, wspace=.14)
    for ax, role, title in zip(axes, ["CERT", "TEST"], ["(a) CERT (validation)", "(b) TEST"]):
        clean_axis(ax, fixed["gamma"])
        ax.set_title(title, loc="left", weight="bold", pad=10)
        for index, report in enumerate(reports):
            y = len(reports)-1-index
            if report["analysis_role"] == "primary":
                ax.axhspan(y-.40, y+.40, facecolor="#F1F5F9", zorder=0)
            overall = report["splits"][role]["bootstrap"]["overall"]
            dot_interval(ax, overall["individual_max_tv"], y+.11,
                         overall["individual_simultaneous_interval"], BLUE)
            dot_interval(ax, overall["joint_max_tv"], y-.11,
                         overall["joint_simultaneous_interval"], ORANGE)
            if role == "CERT":
                ax.plot(PUBLISHED_CERT[report["metadata"]["split_manifest_seed"]], y-.11,
                        marker="D", markersize=6, markerfacecolor="none", markeredgewidth=1.2,
                        color="#222222", linestyle="none", zorder=4)
        ax.set_xlim(.125, .242)
        ax.set_xticks([.14, .18, .20, .22, .24])
        ax.set_ylim(-.55, 4.55)
        ax.set_xlabel("Worst endpoint/context mean TV")
        ax.text(.202, .98, r"$\gamma=0.20$", transform=ax.get_xaxis_transform(),
                va="top", fontsize=9, color=GRAY)
    axes[0].set_yticks(range(4, -1, -1), [str(r["metadata"]["split_manifest_seed"])
                                         + (" *" if r["analysis_role"] == "primary" else "")
                                         for r in reports])
    axes[0].set_ylabel("Grouped split seed")
    handles = [Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Individual maximum"),
               Line2D([], [], color=ORANGE, marker="o", linestyle="none", label="Joint-refit maximum"),
               Line2D([], [], color="#222222", marker="D", markerfacecolor="none", linestyle="none",
                      label="Paper CERT point (3 d.p.)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .97),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.5)
    fig.text(.02, .045,
             "Fixed removal: J07/J08/J11. * Primary split; others are overlapping sensitivity checks, not independent replications.\n"
             f"95% bands are conditional on FIT weights and simultaneous within each role/split; "
             f"{repetitions:,} grouped bootstraps.\n"
             "No simultaneous coverage across the five partitions or CERT/TEST roles is claimed.",
             fontsize=8.5, va="bottom", color="#333333")
    save(fig, output, "fixed_trio_five_splits")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path,
                        default=REAL / "core20/exp01_dependency_significance/results.json")
    parser.add_argument("--fixed", type=Path, default=REAL / "core20_fixed_case/results.json")
    parser.add_argument("--responses", type=Path, default=REAL / "core20/responses.npz",
                        help="Response artifact; its SHA-256 must match both reports")
    parser.add_argument("--split-manifest-dir", type=Path, default=ROOT / "data",
                        help="Directory containing the original split-manifest basenames; hashes remain mandatory")
    parser.add_argument("--output", type=Path, default=REAL / "core20/figures_dependency")
    args = parser.parse_args()
    core = json.loads(args.main.read_text())
    fixed = json.loads(args.fixed.read_text())
    if core["status"] != "complete" or fixed["status"] != "complete" or len(fixed["split_reports"]) != 5:
        raise ValueError("Both analyses must be complete, including all five fixed split reports")
    if core["removed_judge_ids"] != ["J07", "J08", "J11"] or fixed["primary_index"] != 0:
        raise ValueError("Figure layout is for the declared fixed trio and first primary split")
    if core["gamma"] != .20 or fixed["gamma"] != .20:
        raise ValueError("This figure set documents the predeclared gamma=.20 audit")
    all_fixed_bootstraps = [r["splits"][role]["bootstrap"] for r in fixed["split_reports"]
                           for role in ("CERT", "TEST")]
    if core["uncertainty"]["alpha"] != .05 or any(b["alpha"] != .05 for b in all_fixed_bootstraps):
        raise ValueError("These figures label 95% bands and require alpha=.05")
    if len({b["repetitions"] for b in all_fixed_bootstraps}) != 1:
        raise ValueError("The fixed-case figure requires matching bootstrap counts across roles/splits")
    if core["metadata"]["input_sha256"] != fixed["input_sha256"]:
        raise ValueError("Main and fixed-case analyses must use the same response cache")
    source_paths = {"main_results": args.main, "fixed_results": args.fixed,
                    "responses": args.responses, "plot_script": Path(__file__)}
    for report in fixed["split_reports"]:
        seed = report["metadata"]["split_manifest_seed"]
        if seed not in PUBLISHED_CERT:
            raise ValueError("Unexpected split seed for the published Figure 2a comparison")
        source_paths[f"split_manifest_{seed}"] = (
            args.split_manifest_dir / Path(report["metadata"]["split_manifest"]).name)
    paper = ROOT / "supplementary/paper.txt"
    if paper.exists():
        source_paths["paper_extracted_text"] = paper
    captions = here / "README.md"
    if captions.exists():
        source_paths["captions_and_usage"] = captions
    initial_hashes = {name: digest(path) for name, path in source_paths.items()}
    if initial_hashes["responses"] != fixed["input_sha256"]:
        raise ValueError("Response artifact hash no longer matches analysis provenance")
    for report in fixed["split_reports"]:
        seed = report["metadata"]["split_manifest_seed"]
        if initial_hashes[f"split_manifest_{seed}"] != report["metadata"]["split_manifest_sha256"]:
            raise ValueError("Split manifest hash mismatch")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": True, "savefig.pad_inches": .12})
    args.output.mkdir(parents=True, exist_ok=True)
    main_figure(core, args.output)
    composition_figure(core, args.output)
    five_split_figure(fixed, args.output)
    for name, path in source_paths.items():
        if digest(path) != initial_hashes[name]:
            raise RuntimeError(f"Source changed during rendering: {name}")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Render frozen results only; no inference, refitting or new resampling",
        "sources": {name: {"path": str(path.resolve()), "sha256": initial_hashes[name]}
                    for name, path in source_paths.items()},
        "published_values": {"source": "Paper Figure 2a, CERT validation points printed to 3 decimals",
                             "by_split_seed": PUBLISHED_CERT,
                             "note": "These are reported points, not raw original-cache values or confidence bounds"},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "artifacts": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                      for p in sorted(args.output.iterdir())
                      if p.suffix in {".png", ".pdf", ".svg", ".csv"}},
    }
    (args.output / "source_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered 3 figures × PNG/PDF/SVG and 4 CSVs in {args.output}")


if __name__ == "__main__":
    main()
