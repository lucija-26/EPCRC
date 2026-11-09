"""Tables and figures for E9, generated only from results/joint_removal_audit/*.

Writes figures/ (A-D as PDF+PNG) and TABLES.md, which RESULTS.md and
REBUTTAL_EVIDENCE.md quote.

    python experiments/report_e9_joint_removal.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "joint_removal_audit")
FIG = os.path.join(RES, "figures")
GAMMA = 0.20
PRIMARY = 20260818
TRIPLE = "J07+J08+J11"
ORDER = ["J07", "J08", "J11", "J07+J08", "J07+J11", "J08+J11", TRIPLE]
TARGETS = ["J07", "J08", "J11"]


def f4(x):
    return f"{x:.4f}"


def ci(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    cov = pd.read_csv(os.path.join(RES, "coverage_errors.csv"))
    ep = pd.read_csv(os.path.join(RES, "endpoint_errors.csv"))
    boot = pd.read_csv(os.path.join(RES, "bootstrap_statistics.csv"))
    comp = pd.read_csv(os.path.join(RES, "composition_metrics.csv"))
    compep = pd.read_csv(os.path.join(RES, "composition_endpoint_metrics.csv"))
    edges = pd.read_csv(os.path.join(RES, "dependency_edges.csv"))
    coords = pd.read_csv(os.path.join(RES, "coordinate_deltas.csv"))
    bracket_path = os.path.join(RES, "witness_bracket.csv")
    import json
    bracket = (pd.read_csv(bracket_path) if os.path.exists(bracket_path) and os.path.getsize(bracket_path) > 1
               else pd.DataFrame())
    seeds = sorted(cov.split_seed.unique())
    out = []

    # ---- Table 1: individual vs triple, per split, V and T ----
    import json
    tx = json.load(open(os.path.join(RES, "test_exposure.json")))
    t_role = ("fixed-design inferential check; "
              + (f"note: E1 backward elimination already reported T={tx['e1_primary_split_TEST_k17']:.4f} for this exact deletion at k=17"
                 if tx["e1_backward_k17_is_this_triple"] else "T not previously reported for this deletion"))
    for q, role in (("V", "descriptive: V selected the trio"), ("T", t_role)):
        out.append(f"\n### Table 1{q}. Individual vs triple refit, partition {q} ({role})\n")
        out.append("| Split | Target | Individual | Triple refit | Increase | Paired 95% CI | Simultaneous band (5 splits) | Margin vs 0.20 | Margin band |")
        out.append("|---|---|---:|---:|---:|---|---|---:|---|")
        for s in seeds:
            b = boot[(boot.split_seed == s) & (boot.partition == q) & (boot.removal_set == TRIPLE)
                     & (boot.comparison == "individual_vs_refit")]
            for tgt in TARGETS + ["ALL"]:
                r = b[b.target == tgt].iloc[0]
                name = "**max over trio**" if tgt == "ALL" else tgt
                out.append(f"| {s} | {name} | {f4(r.base_error)} | {f4(r.other_error)} | {r.delta:+.4f} | "
                           f"{ci(r.delta_ci_low, r.delta_ci_high)} | {ci(r.band_delta_low, r.band_delta_high)} | "
                           f"{r.margin:+.4f} | {ci(r.band_margin_low, r.band_margin_high)} |")

    out.append("\n### Table 1-adj. Bonferroni-adjusted 99% percentile interval for each split's aggregate increase (family = 5 splits, separately for V and T)\n")
    out.append("| Split | Q | increase | 95% CI | Bonferroni 99% CI |")
    out.append("|---|---|---:|---|---|")
    for s in seeds:
        for q in ("V", "T"):
            r = boot[(boot.split_seed == s) & (boot.partition == q) & (boot.removal_set == TRIPLE)
                     & (boot.comparison == "individual_vs_refit") & (boot.target == "ALL")].iloc[0]
            assert r.family_size == 5
            out.append(f"| {s} | {q} | {r.delta:+.4f} | {ci(r.delta_ci_low, r.delta_ci_high)} | {ci(r.delta_adj_ci_low, r.delta_adj_ci_high)} |")

    out.append("\n### Table 1-pairs. Pairwise removals, pointwise 95% intervals (exploratory, no multiplicity adjustment)\n")
    out.append("| Split | Q | pair | max individual | pair refit | increase | 95% CI | margin vs 0.20 | 95% CI of pair error |")
    out.append("|---|---|---|---:|---:|---:|---|---:|---|")
    for s in seeds:
        for q in ("V", "T"):
            for R in ORDER[3:6]:
                r = boot[(boot.split_seed == s) & (boot.partition == q) & (boot.removal_set == R)
                         & (boot.comparison == "individual_vs_refit") & (boot.target == "ALL")].iloc[0]
                out.append(f"| {s} | {q} | {R} | {f4(r.base_error)} | {f4(r.other_error)} | {r.delta:+.4f} | "
                           f"{ci(r.delta_ci_low, r.delta_ci_high)} | {r.margin:+.4f} | [{r.other_ci_low:.4f}, {r.other_ci_high:.4f}] |")

    out.append("\n### Table 3-ci. Composed map minus direct refit, paired 95% intervals (supporting)\n")
    out.append("| Split | Q | removal set | refit | composed | composed - refit | 95% CI |")
    out.append("|---|---|---|---:|---:|---:|---|")
    for s in seeds:
        for q in ("V", "T"):
            for R in ORDER[3:]:
                r = boot[(boot.split_seed == s) & (boot.partition == q) & (boot.removal_set == R)
                         & (boot.comparison == "refit_vs_comp") & (boot.target == "ALL")]
                if len(r):
                    r = r.iloc[0]
                    out.append(f"| {s} | {q} | {R} | {f4(r.base_error)} | {f4(r.other_error)} | {r.delta:+.4f} | {ci(r.delta_ci_low, r.delta_ci_high)} |")

    # ---- Table 2: all seven removal sets ----
    for q in ("F", "V", "T"):
        out.append(f"\n### Table 2{q}. All seven removal sets, partition {q} (coverage error = max over removed targets and contexts)\n")
        out.append("| Removal set | " + " | ".join(str(s) for s in seeds) + " | passes 0.20 in |")
        out.append("|---|" + "---:|" * len(seeds) + "---|")
        for R in ORDER:
            vals = [cov[(cov.split_seed == s) & (cov.partition == q) & (cov.removal_set == R)].iloc[0] for s in seeds]
            npass = sum(v.pass_at_gamma for v in vals)
            out.append(f"| {R} | " + " | ".join(f4(v.E_refit) for v in vals) + f" | {npass}/{len(seeds)} |")

    for q in ("V", "T"):
        out.append(f"\n### Table 2{q}-detail. Per removal set and split, partition {q}\n")
        out.append("| Split | Removal set | E_ind (same targets) | E_refit | Delta | margin | pass | worst target | worst context |")
        out.append("|---|---|---:|---:|---:|---:|---|---|---|")
        for s_ in seeds:
            for R in ORDER:
                v = cov[(cov.split_seed == s_) & (cov.partition == q) & (cov.removal_set == R)].iloc[0]
                out.append(f"| {s_} | {R} | {f4(v.E_individual)} | {f4(v.E_refit)} | {v.Delta:+.4f} | {v.margin:+.4f} | "
                           f"{'yes' if v.pass_at_gamma else 'no'} | {v.worst_target} | {v.worst_context} |")

    # ---- Table 3: dependency and composition ----
    out.append("\n### Table 3. Theorem 2 executed: singleton witnesses -> composed survivor-only map -> direct refit\n")
    out.append("| Split | Removal set | rho(A) | max row mass on removed | min direct survivor mass | max E[tau] | status | Q | composed | refit | composed - refit | bound raw | bound min(1,U) | context bound | empirical bound <= 0.20 |")
    out.append("|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for s in seeds:
        for R in ORDER[3:]:
            for q in ("F", "V", "T"):
                r = comp[(comp.split_seed == s) & (comp.removal_set == R) & (comp.partition == q)].iloc[0]
                if r.status != "ok":
                    out.append(f"| {s} | {R} | {r.rho_A:.4f} | | | {r.status} | {q} | | | | | |")
                    continue
                out.append(f"| {s} | {R} | {r.rho_A:.4f} | {r.max_dependency_mass:.4f} | {r.min_survivor_mass:.4f} | {r.max_expected_depth:.3f} | ok | {q} | "
                           f"{f4(r.E_comp)} | {f4(r.E_refit)} | {r.comp_minus_refit:+.4f} | {f4(r.bound_raw)} | {f4(r.bound_clipped)} | {f4(r.bound_context)} | "
                           f"{'yes' if r.bound_certifies_gamma else 'no'} |")
    out.append("\n### Table 3-appendix. Per-target composition (eps = singleton residual on Q, bound row = (H eps)_i)\n")
    out.append("| Split | Removal set | Q | target | eps (singleton) | composed | refit | bound row | context bound row | E[tau] | survivor mass |")
    out.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in compep.iterrows():
        out.append(f"| {r.split_seed} | {r.removal_set} | {r.partition} | {r.target} | {f4(r.epsilon_singleton)} | {f4(r.comp_error)} | "
                   f"{f4(r.refit_error)} | {f4(r.bound)} | {f4(r.bound_context)} | {r.expected_depth:.3f} | {r.survivor_mass:.4f} |")

    # ---- Table 4: dependency edges, primary split ----
    out.append(f"\n### Table 4. Singleton witness weights among the trio (rows = target, columns = donor), split {PRIMARY}\n")
    e = edges[(edges.split_seed == PRIMARY) & (edges.removal_set == TRIPLE)]
    out.append("| target \\ donor | J07 | J08 | J11 | mass on the 17 survivors |")
    out.append("|---|---:|---:|---:|---:|")
    for t in TARGETS:
        row = {u: (e[(e.target == t) & (e.donor == u)].weight.iloc[0] if u != t else 0.0) for u in TARGETS}
        out.append(f"| {t} | " + " | ".join(f"{row[u]:.4f}" for u in TARGETS) + f" | {1 - sum(row.values()):.4f} |")
    out.append("\nAll splits (triple rows):\n")
    out.append("| Split | J07->J08 | J07->J11 | J08->J07 | J08->J11 | J11->J07 | J11->J08 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in seeds:
        e = edges[(edges.split_seed == s) & (edges.removal_set == TRIPLE)]
        vals = [e[(e.target == a) & (e.donor == b)].weight.iloc[0]
                for a, b in [("J07", "J08"), ("J07", "J11"), ("J08", "J07"), ("J08", "J11"), ("J11", "J07"), ("J11", "J08")]]
        out.append(f"| {s} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")

    # ---- Table 5: near-tie diagnostics + fixed coordinate test ----
    out.append("\n### Table 5. Near-tie diagnostics and the fixed-coordinate check (triple, individual vs refit)\n")
    out.append("| Split | Q | argmax stability (joint) | bootstrap bias of joint max | fixed coordinate (from V) | its increase on Q | percentile 95% CI | max-t simultaneous band (21 coords) |")
    out.append("|---|---|---:|---:|---|---:|---|---|")
    for s in seeds:
        for q in ("V", "T"):
            r = boot[(boot.split_seed == s) & (boot.partition == q) & (boot.removal_set == TRIPLE)
                     & (boot.comparison == "individual_vs_refit") & (boot.target == "ALL")].iloc[0]
            c = coords[(coords.split_seed == s) & (coords.partition == q) & (coords.fixed_coordinate_from_V)].iloc[0]
            out.append(f"| {s} | {q} | {r.argmax_stability_other:.3f} | {r.boot_bias_other:+.4f} | {c.target} / {c.context} | "
                       f"{c.delta:+.4f} | {ci(c.ci_low, c.ci_high)} | {ci(c.maxt_band_low, c.maxt_band_high)} |")
    out.append("\nCoordinates whose max-t band excludes zero (out of 21 per split):\n")
    out.append("| Split | Q | positive | negative |")
    out.append("|---|---|---:|---:|")
    for s in seeds:
        for q in ("V", "T"):
            c = coords[(coords.split_seed == s) & (coords.partition == q)]
            out.append(f"| {s} | {q} | {(c.maxt_band_low > 0).sum()} | {(c.maxt_band_high < 0).sum()} |")

    # ---- Table 6: witness bracket ----
    if len(bracket):
        out.append("\n### Table 6. Solver dependence of the singleton witnesses: range of row mass on the other removed judges over all F-witnesses within `slack` of the exact optimum\n")
        rm = bracket[bracket.kind == "row_mass"]
        fits = json.load(open(os.path.join(RES, "fit_records.json")))
        gap = {(f["split_seed"], f["target"]): f["fit_objective_F"] for f in fits if f["removal_set"] == f["target"]}
        out.append("The original witness comes from the cutting-plane fitter, whose F objective can sit slightly above the exact optimum; "
                   "it lies inside the bracket only when that gap is <= slack.\n")
        out.append("| Split | slack | removal set | target | original witness gap above exact opt | original | min | max | original inside | min/max fits within slack |")
        out.append("|---|---:|---|---|---:|---:|---:|---:|---|---|")
        for _, r in rm.iterrows():
            g = gap[(r.split_seed, r.target)] - r.exact_opt_F
            inside = r.mass_min - 1e-9 <= r.mass_original <= r.mass_max + 1e-9
            within = max(r.fit_err_at_min, r.fit_err_at_max) <= r.exact_opt_F + r.slack + 1e-7
            out.append(f"| {r.split_seed} | {r.slack:g} | {r.removal_set} | {r.target} | {g:.2e} | {r.mass_original:.4f} | "
                       f"{r.mass_min:.4f} | {r.mass_max:.4f} | {'yes' if inside else 'no (gap > slack)' if g > r.slack else 'NO'} | {'yes' if within else 'NO'} |")
        cw = bracket[bracket.kind != "row_mass"]
        out.append("\nComposition rerun at the two extreme witnesses (illustration, not a bound):\n")
        out.append("| Split | slack | removal set | witness | rho(A) | composed F | composed V | composed T | bound V |")
        out.append("|---|---:|---|---|---:|---:|---:|---:|---:|")
        for _, r in cw.iterrows():
            if r.status == "ok":
                out.append(f"| {r.split_seed} | {r.slack:g} | {r.removal_set} | {r.kind.replace('composition_at_', '')} | {r.rho_A:.4f} | "
                           f"{f4(r.E_comp_F)} | {f4(r.E_comp_V)} | {f4(r.E_comp_T)} | {f4(r.bound_V)} |")
            else:
                out.append(f"| {r.split_seed} | {r.slack:g} | {r.removal_set} | {r.kind} | {r.rho_A:.4f} | {r.status} | | | |")

    with open(os.path.join(RES, "TABLES.md"), "w") as f:
        f.write("# E9 tables (generated by experiments/report_e9_joint_removal.py)\n" + "\n".join(out) + "\n")

    # ---- Figure A: all removal sets, V ----
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(ORDER))
    for k, s in enumerate(seeds):
        y = [cov[(cov.split_seed == s) & (cov.partition == "V") & (cov.removal_set == R)].E_refit.iloc[0] for R in ORDER]
        ax.plot(x + (k - (len(seeds) - 1) / 2) * 0.07, y, "o", ms=5, label=str(s))
    ax.axhline(GAMMA, color="k", ls="--", lw=1)
    ax.text(len(ORDER) - 0.5, GAMMA + 0.002, "gamma = 0.20", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, [r.replace("+", "+\n") for r in ORDER], fontsize=8)
    ax.set_ylabel("validation coverage error")
    ax.set_title("Figure A. Coverage error after each removal set (V, refit on F), five splits", fontsize=9)
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    save(fig, "A_all_removal_sets_V")

    # ---- Figure B: individual vs triple by target, primary split, V and T ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, q in zip(axes, ("V", "T")):
        b = boot[(boot.split_seed == PRIMARY) & (boot.partition == q) & (boot.removal_set == TRIPLE)
                 & (boot.comparison == "individual_vs_refit")]
        for k, t in enumerate(TARGETS):
            r = b[b.target == t].iloc[0]
            ax.plot([k - 0.12, k + 0.12], [r.base_error, r.other_error], "-", color="grey", lw=1)
            ax.plot(k - 0.12, r.base_error, "o", color="tab:blue", label="individual" if k == 0 else None)
            ax.plot(k + 0.12, r.other_error, "s", color="tab:red", label="triple refit" if k == 0 else None)
            ax.text(k, r.other_error + 0.004, f"{r.delta:+.3f}\n[{r.delta_ci_low:+.3f}, {r.delta_ci_high:+.3f}]",
                    ha="center", fontsize=7)
        ax.axhline(GAMMA, color="k", ls="--", lw=1)
        ax.set_xticks(range(3), TARGETS)
        ax.set_title(f"partition {q}", fontsize=9)
    axes[0].set_ylabel("worst-context mean TV")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Figure B. Per-target error, split {PRIMARY}; labels: increase and paired 95% CI", fontsize=9)
    save(fig, "B_individual_vs_triple")

    # ---- Figure C: dependency matrix, primary split ----
    e = edges[(edges.split_seed == PRIMARY) & (edges.removal_set == TRIPLE)]
    mat = np.zeros((3, 4))
    for a, t in enumerate(TARGETS):
        for b_, u in enumerate(TARGETS):
            if t != u:
                mat[a, b_] = e[(e.target == t) & (e.donor == u)].weight.iloc[0]
        mat[a, 3] = 1 - mat[a, :3].sum()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    for a in range(3):
        for b_ in range(4):
            ax.text(b_, a, f"{mat[a, b_]:.2f}", ha="center", va="center",
                    color="white" if mat[a, b_] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(4), TARGETS + ["17 survivors"], fontsize=8)
    ax.set_yticks(range(3), TARGETS, fontsize=8)
    ax.set_xlabel("donor")
    ax.set_ylabel("target")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Figure C. Singleton witness weights A | B mass, split {PRIMARY}", fontsize=8)
    save(fig, "C_dependency_matrix")

    # ---- Figure D: composition vs refit vs bound, triple, V and T ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, q in zip(axes, ("V", "T")):
        for k, s in enumerate(seeds):
            r = comp[(comp.split_seed == s) & (comp.removal_set == TRIPLE) & (comp.partition == q)].iloc[0]
            if r.status != "ok":
                ax.text(k, 0.25, r.status, rotation=90, fontsize=6)
                continue
            ax.plot(k, r.E_individual, "o", color="tab:blue", label="max individual" if k == 0 else None)
            ax.plot(k, r.E_refit, "s", color="tab:red", label="direct refit" if k == 0 else None)
            ax.plot(k, r.E_comp, "D", color="tab:orange", label="composed M (no refit)" if k == 0 else None)
            ax.plot(k, r.bound_clipped, "v", color="tab:grey", label="Theorem 2 bound" if k == 0 else None)
        ax.axhline(GAMMA, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(seeds)), [str(s)[-2:] for s in seeds])
        ax.set_xlabel("split seed (last digits)")
        ax.set_title(f"partition {q}", fontsize=9)
    axes[0].set_ylabel("coverage error, triple removal")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Figure D. Bound >= composed map (upper bound for M, not a lower bound for the refit)", fontsize=9)
    save(fig, "D_composition_vs_refit")
    print("wrote", os.path.join(RES, "TABLES.md"), "and", FIG)


if __name__ == "__main__":
    sys.exit(main())
