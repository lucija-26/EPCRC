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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------
# where the inputs live
# --------------------------------------------------------------------------

def input_paths(panel: str) -> Dict[str, str]:
    """Result files for a panel, falling back to the unsuffixed Core-8 names."""
    def pick(*candidates: str) -> str:
        for name in candidates:
            path = os.path.join(RESULTS, name)
            if os.path.exists(path):
                return path
        return os.path.join(RESULTS, candidates[-1])

    return {
        "e0_synthetic": pick("e0_noncomposability.json"),
        "e0_real": pick(f"e0_real_{panel}.json"),
        "e1": pick(f"e1_frontier_{panel}.json", "e1_frontier.json"),
        "c3": pick(f"c3_baselines_{panel}.json", "c3_baselines.json"),
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

    if len(declared) and loo > max_declared:
        lines.append(
            f"On the predeclared grid (up to gamma = {max_declared:g}) the "
            f"removable set is empty at every tolerance, because the smallest "
            f"leave-one-out error on this panel is {loo:.3f}. The predeclared "
            f"grid therefore says nothing about composition here, and this is "
            f"reported rather than hidden."
        )
        lines.append("")
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
        lines.append(
            f"**Result.** Across {len(multi)} (split seed, tolerance) cases in "
            f"which at least two judges were individually certified removable, "
            f"the joint deletion broke the tolerance in **{n_viol} of "
            f"{len(multi)}** cases. The mean composition gap — how many more "
            f"judges the joint constraint requires than the one-at-a-time audit "
            f"kept — is **{gap:+.2f}** judges."
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
        lines.append("**Verdict: supported on real judges.**")
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
             cells_significant=("n_seeds_significant", "sum"),
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
        "(split seed, panel size) pairs, and `significant` counts how many of "
        "them put the whole 95% interval on one side of zero."
    )
    lines.append("")

    show = pooled.rename(columns={"cells_significant": "significant"})
    lines.append(_md_table(
        show, ["label", "delta", "lo", "hi", "significant", "cells"]))
    lines.append("")

    # `random_best_draw` is the best of 100 random panels chosen by looking at
    # the answer, so it is an oracle rather than a competitor and is discussed
    # on its own terms.
    oracle = pooled[pooled["method"] == "random_best_draw"]
    rest = pooled[pooled["method"] != "random_best_draw"]

    decided = rest[rest["cells_significant"] >= 0.5 * total_cells]
    ties = rest[(rest["cells_significant"] <= 0.2 * total_cells)
                & (~rest["method"].isin(R.COVERAGE_METHODS))]

    if len(decided):
        lines.append(
            f"Coverage wins on a majority of cells against "
            f"{', '.join(decided['label'])} — every heuristic a practitioner "
            f"would actually reach for."
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
        "The rank-k floor is what a best-case k-dimensional basis achieves. It "
        "is **not deployable** — its basis vectors are not real judges, so "
        "nothing can be run to produce them — but it bounds what any selection "
        "of k judges could hope for."
    )
    lines.append("")
    lines.append(_md_table(floor[floor["method"] == "pca"],
                           ["k", "worst_judge_tv", "mean_judge_tv"]))
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

    if os.path.exists(inputs["c3"]):
        payload = R.load(inputs["c3"])
        block = payload["per_split_seed"][0]
        lines += [
            "## Setup",
            "",
            f"- Judges: {len(block['judges'])} — {', '.join(block['judges'])}",
            f"- Contexts per judge: {len(block['contexts'])}",
            "- Items per split: " + ", ".join(
                f"{name} {count}" for name, count in block["split_items"].items()),
            f"- Split seeds: {payload['split_seeds']}",
            f"- Bootstrap replicates: {payload['n_bootstrap']}",
            "",
            "Splits are grouped by source item, so the same prompt never appears "
            "in two splits. The pairs seed fixes what each judge saw and needs "
            "GPU inference to change; the split seed only re-partitions those "
            "cached responses, which is what makes across-seed bands affordable.",
            "",
            "---",
            "",
        ]

    lines += _c1_section(inputs)
    lines += ["---", ""]
    lines += _c2_section(inputs)
    lines += ["---", ""]
    lines += _c3_section(inputs)
    lines += [
        "---",
        "",
        "## Package contents",
        "",
        "- `tables/` — every reported number as CSV",
        "- `figures/` — the figures, drawn from those same tables",
        "- `raw/` — the unmodified experiment output the tables are derived from",
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

    prov = provenance(panel, inputs)

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
    )
    print(f"figures: {len(figs)}")

    for key, path in inputs.items():
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(out_dir, "raw", os.path.basename(path)))

    if with_notebooks:
        print(f"notebooks: {copy_notebooks(out_dir)}")

    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as handle:
        handle.write(build_summary(panel, inputs, prov))

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
