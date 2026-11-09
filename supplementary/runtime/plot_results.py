"""Plot completed real exp02/04/05 results without refitting or inventing CIs.

Example:
  .venv/bin/python supplementary/runtime/plot_results.py \
    --run-dir results/supplementary/real/nli4 \
    --output-dir results/supplementary/real/nli4/figures
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

COLORS = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
          "purple": "#8C62A8", "red": "#B44660", "gray": "#A8ADB4"}
FILES = {"reconstruction": "exp02_reconstruction/results.json",
         "selection": "exp04_selection_benchmarks/results.json",
         "cost": "exp05_measured_cost/results.json"}


def read_sources(run_dir):
    sources, hashes = {}, {}
    for name, relative in FILES.items():
        path = run_dir / relative
        raw = path.read_bytes()
        result = json.loads(raw)
        if result.get("metadata", {}).get("synthetic") is not False:
            raise ValueError(f"Refusing unverified or synthetic result source: {path}")
        sources[name] = result
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    response_hashes = {s["metadata"].get("input_sha256") for s in sources.values()}
    if len(response_hashes) != 1 or None in response_hashes:
        raise ValueError("The three experiment results must identify the same response-cache SHA-256")
    return sources, hashes


def descriptions(run_dir, sources):
    reconstruction = sources["reconstruction"]
    metadata = reconstruction["metadata"]
    provenance = metadata["input_provenance"]
    n_models, n_contexts = len(reconstruction["judge_ids"]), len(reconstruction["contexts"])
    if metadata.get("interface") == "nli" and provenance.get("config") == "mnli":
        name = f"NLI{n_models} · MNLI {provenance['split'].replace('validation_', '')} validation"
    else:
        name = metadata.get("ecosystem", "Model ecosystem").replace("_", " ")
        name += " · " + provenance.get("dataset_id", "declared dataset").split("/")[-1]
    first = reconstruction["budgets"][0]
    losses_file = run_dir / "exp02_reconstruction" / first["test_losses_file"]
    with np.load(losses_file, allow_pickle=False) as archive:
        groups = archive["test_base_item_groups"]
    subtitle = (f"{n_models} models · {n_contexts} context{'s' if n_contexts != 1 else ''} · "
                f"held-out TEST: {len(groups):,} pairs / {len(np.unique(groups)):,} source groups")
    return name, subtitle


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .8,
                         "axes.edgecolor": "#50555D", "xtick.color": "#34383E",
                         "ytick.color": "#34383E", "text.color": "#202329",
                         "legend.fontsize": 9, "legend.frameon": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})


def axes_style(ax, ks=None):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", linewidth=.65)
    if ks is not None:
        ax.set_xticks(ks)
        ax.set_xlim(min(ks) - .15, max(ks) + .15)
        ax.set_xlabel("Retained models, k")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def equal_series(a, b):
    return np.shape(a) == np.shape(b) and np.allclose(a, b, atol=1e-12, rtol=0)


def reconstruction_plot(ax, source):
    rows = sorted(source["budgets"], key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    values = {method: [r["methods"][method]["test"]["worst_endpoint_context_tv"] for r in rows]
              for method in rows[0]["methods"]}
    same = equal_series(values["global_convex"], values["context_convex"])
    labels = {"global_convex": "Global / context convex" if same else "Global convex",
              "context_convex": "Context-specific convex", "temperature_convex": "Temperature calibration",
              "nonlinear_gate": "Nonlinear gate"}
    specs = [("global_convex", COLORS["blue"], "o", "-", 2.0, 6.5),
             ("context_convex", COLORS["purple"], "D", ":", 1.5, 5.0),
             ("temperature_convex", COLORS["orange"], "^", "--", 1.6, 6.5),
             ("nonlinear_gate", COLORS["green"], "s", "-.", 1.4, 4.5)]
    for method, color, marker, linestyle, width, size in specs:
        if method == "context_convex" and same:
            continue
        ax.plot(ks, values[method], label=labels[method], color=color, marker=marker,
                linestyle=linestyle, linewidth=width, markersize=size,
                markerfacecolor="white", markeredgewidth=1.2)
    axes_style(ax, ks)
    ax.set_ylabel("TEST worst endpoint/context mean TV")
    ax.set_ylim(-.004, max(max(v) for v in values.values()) * 1.18)
    ax.set_title("(a) Replacement maps on shared supports", loc="left", pad=12)
    ax.legend(loc="upper right")
    note = "FIT-selected shared supports; CERT tunes map parameters."
    if same and len(source["contexts"]) == 1:
        note += "\nOne context: global and context-specific curves coincide."
    elif same:
        note += "\nGlobal and context-specific curves coincide on these budgets."
    note += "\nPoint estimates; no confidence intervals drawn."
    selected_trials = [diagnostic["trials"][diagnostic["selected_trial"]]
                       for row in rows
                       for diagnostic in row["methods"]["nonlinear_gate"].get("fit_diagnostics", {}).values()]
    unconverged = sum(not trial["converged"] for trial in selected_trials)
    if unconverged:
        note += (f"\nGate: {unconverged}/{len(selected_trials)} selected fits unconverged "
                 f"at maxiter={source['configuration']['maxiter']}; see iteration audit.")
    return note


def selection_plot(ax, source):
    rows = source["rows"]
    ks = sorted({r["k"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    values = {method: [next(r["test_tv"] for r in rows if r["method"] == method and r["k"] == k)
                       for k in ks] for method in methods}
    random = [method for method in methods if method.startswith("random_")]
    for method in random:
        ax.plot(ks, values[method], color=COLORS["gray"], linewidth=.8, linestyle=":",
                marker=".", markersize=4, alpha=.7, zorder=1)
    labels = {"backward": "Backward coverage", "forward": "Forward coverage", "qr": "Pivoted QR",
              "kmedoids": "k-medoids", "swap1": "1-swap", "swap2": "2-swap"}
    merge_swaps = "swap1" in values and "swap2" in values and equal_series(values["swap1"], values["swap2"])
    if merge_swaps:
        labels["swap1"] = "1-swap / 2-swap"
    specs = [("backward", COLORS["blue"], "o", "-", 2.0, 7),
             ("forward", COLORS["orange"], "^", "--", 1.5, 6),
             ("qr", COLORS["purple"], "D", ":", 1.5, 5),
             ("kmedoids", COLORS["green"], "s", "-.", 1.2, 4.5),
             ("swap1", COLORS["red"], "+", "--", 1.2, 9),
             ("swap2", "#7A592D", "x", ":", 1.2, 7)]
    for method, color, marker, linestyle, width, size in specs:
        if method not in values or (method == "swap2" and merge_swaps):
            continue
        ax.plot(ks, values[method], label=labels[method], color=color, marker=marker,
                linestyle=linestyle, linewidth=width, markersize=size,
                markerfacecolor="white", markeredgewidth=1.2, zorder=3)
    handles, legend_labels = ax.get_legend_handles_labels()
    if random:
        handles.append(Line2D([], [], color=COLORS["gray"], linestyle=":", marker="."))
        legend_labels.append(f"Random ({len(random)} draws)")
    ax.legend(handles, legend_labels, loc="upper right", ncol=2, columnspacing=.8, handlelength=2)
    axes_style(ax, ks)
    ax.set_ylabel("TEST worst endpoint/context mean TV")
    ax.set_ylim(-.004, max(max(v) for v in values.values()) * 1.35)
    ax.set_title("(b) Selection at equal model budgets", loc="left", pad=12)
    return ("All methods use global convex reconstruction fitted on FIT.\n"
            "Gray traces show individual random draws, not uncertainty bands.\n"
            "Exact FIT optimality does not imply TEST optimality.")


def cost_plot(ax, source):
    rows = source["rows"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    groups = {method: sorted((r for r in rows if r["method"] == method), key=lambda r: -r["k"])
              for method in methods}
    def coordinates(method):
        return np.asarray([[r["test_worst_endpoint_context_tv"],
                            100 * r["endpoint_service_work_saving_fraction"]] for r in groups[method]])
    merge_cost = "qr" in methods and "cost_aware" in methods and equal_series(coordinates("qr"), coordinates("cost_aware"))
    labels = {"backward": "Backward coverage", "qr": "QR / cost-aware" if merge_cost else "Pivoted QR",
              "cheapest": "Cheapest endpoints", "cost_aware": "Cost-aware coverage"}
    specs = [("backward", COLORS["blue"], "o", "-"),
             ("qr", COLORS["purple"], "D", ":"),
             ("cheapest", COLORS["orange"], "^", "--"),
             ("cost_aware", COLORS["green"], "s", "-.")]
    labeled = set()
    for method, color, marker, linestyle in specs:
        if method not in groups or (method == "cost_aware" and merge_cost):
            continue
        xy = coordinates(method)
        ax.plot(xy[:, 0], xy[:, 1], label=labels[method], color=color, marker=marker,
                linestyle=linestyle, linewidth=1.7, markersize=6, markerfacecolor="white", markeredgewidth=1.2)
        for row, (x, y) in zip(groups[method], xy):
            key = (round(x, 10), round(y, 10))
            if key in labeled:
                continue
            labeled.add(key)
            label = f"k={row['k']}"
            offset = (6, 8) if y <= 1e-6 else (6, -13)
            if method == "cheapest" and row["k"] == 1:
                offset = (-22, 9)
            elif method == "backward" and row["k"] == 1:
                offset = (5, 8)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8.5, color=color)
    axes_style(ax)
    ax.set_xlabel("TEST worst endpoint/context mean TV")
    ax.set_ylabel("Endpoint service work saved (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(-4, 101)
    ax.set_xlim(-.004, max(r["test_worst_endpoint_context_tv"] for r in rows) * 1.18)
    ax.set_title("(c) Measured service work vs. fidelity", loc="left", pad=12)
    ax.legend(loc="upper left")
    timing = source["timing_provenance"]["per_endpoint_context"]
    device = sorted({r.get("selected_device_name", "declared device") for r in timing})
    batch = sorted({r["batch_size"] for r in timing})
    repeats = sorted({r["repeats"] for r in timing})
    hardware = f"{', '.join(device)}; batch {','.join(map(str, batch))}; {','.join(map(str, repeats))} timing repeats."
    return (hardware + "\nAdditive tokenization + inference + transfer work; not online latency.\n"
            "Savings exclude model loading, downloads, and reconstruction overhead.")


def save_figure(fig, directory, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=300,
                    bbox_inches="tight", pad_inches=.15)


def write_table(output_dir, sources):
    fields = ["figure", "method", "k", "test_worst_endpoint_context_tv",
              "endpoint_service_work_saved_percent", "retained_ids"]
    rows = []
    for budget in sources["reconstruction"]["budgets"]:
        for method, result in budget["methods"].items():
            rows.append({"figure": "reconstruction", "method": method, "k": budget["k"],
                         "test_worst_endpoint_context_tv": result["test"]["worst_endpoint_context_tv"],
                         "retained_ids": ";".join(budget["retained_judge_ids"])})
    for row in sources["selection"]["rows"]:
        rows.append({"figure": "selection", "method": row["method"], "k": row["k"],
                     "test_worst_endpoint_context_tv": row["test_tv"], "retained_ids": ";".join(row["kept_ids"])})
    for row in sources["cost"]["rows"]:
        rows.append({"figure": "measured_service_work", "method": row["method"], "k": row["k"],
                     "test_worst_endpoint_context_tv": row["test_worst_endpoint_context_tv"],
                     "endpoint_service_work_saved_percent": 100 * row["endpoint_service_work_saving_fraction"],
                     "retained_ids": ";".join(row["kept_ids"])})
    with (output_dir / "figure_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    sources, hashes = read_sources(run_dir)
    title, subtitle = descriptions(run_dir, sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    style()
    specs = [("reconstruction", reconstruction_plot, "reconstruction_tv"),
             ("selection", selection_plot, "selection_tv"),
             ("cost", cost_plot, "measured_service_work")]
    captions = {}
    for key, plot, stem in specs:
        fig, ax = plt.subplots(figsize=(7.4, 5.5))
        fig.subplots_adjust(left=.13, right=.97, top=.77, bottom=.24)
        fig.suptitle(title, fontsize=15, fontweight="bold", x=.13, ha="left", y=.98)
        fig.text(.13, .915, subtitle, fontsize=10, color="#4D535C")
        note = plot(ax, sources[key])
        fig.text(.13, .06, note, fontsize=8.6, linespacing=1.55, color="#4D535C")
        save_figure(fig, output_dir, stem)
        plt.close(fig)
        captions[stem] = note
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.8))
    fig.subplots_adjust(left=.055, right=.985, top=.78, bottom=.255, wspace=.30)
    fig.suptitle(title, fontsize=17, fontweight="bold", x=.055, ha="left", y=.975)
    fig.text(.055, .905, subtitle, fontsize=11, color="#4D535C")
    for ax, (key, plot, stem) in zip(axes, specs):
        note = plot(ax, sources[key])
        fig.text(ax.get_position().x0, .065, note, fontsize=8.1, linespacing=1.6, color="#4D535C")
    save_figure(fig, output_dir, "overview")
    plt.close(fig)
    write_table(output_dir, sources)
    manifest = {"source_run_directory": str(run_dir), "sources_sha256": hashes,
                "response_cache_sha256": sources["reconstruction"]["metadata"]["input_sha256"],
                "synthetic": False, "title": title, "subtitle": subtitle,
                "formats": ["png", "pdf", "svg"], "png_dpi": 300,
                "uncertainty": "No CIs or uncertainty bands inferred or plotted. Random traces are the recorded random draws.",
                "captions": captions, "matplotlib_version": matplotlib.__version__}
    (output_dir / "figure_sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved four figures in PNG/PDF/SVG, source manifest and data CSV to {output_dir}")


if __name__ == "__main__":
    main()
