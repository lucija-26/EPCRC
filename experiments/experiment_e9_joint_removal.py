"""Experiment E9 -- individual, pairwise and triple removal of J07, J08, J11.

Follow-up to E0 (protocol 1.0, 2026-11-08).  E0 found three judges that each pass the leave-one-out audit
at gamma = 0.20 on CERT but fail when deleted together.  This experiment
measures the whole transition:

  * Arm A, individual: fit each target on J \\ {i}.
  * Arm B, joint refit: for every pair and the triple R, refit each removed
    target on J \\ R.  In a pair the third candidate is still a donor.
  * Arm C, composition: take the singleton rows, split them into A (weights on
    other removed judges) and B (weights on survivors), and use Theorem 2's
    M = (I - A)^{-1} B with no new fitting.  Score M and the bound H eps.

All weights are fitted on FIT and frozen.  CERT (V) reproduces Figure 2(a);
TEST (T) is the fixed-design check of the same trio.  Uncertainty is a paired
source-task bootstrap with fixed weights.  A bracket of the singleton
witnesses' dependency mass over near-optimal solutions shows how much the
composition arm depends on solver tie-breaking.

No LLM inference.  Reads results/core20/scores and data/split_seed*.json and
writes results/joint_removal_audit/.

Usage:

    python -u experiments/experiment_e9_joint_removal.py              # full run
    python -u experiments/experiment_e9_joint_removal.py --boot 499 --seeds 20260818 --bracket-seeds --lp-seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scipy  # noqa: E402

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.judge import solve_minimax_weights, solve_minimax_weights_lp  # noqa: E402
from epcrc.panel import CORE20, SPLIT_SEEDS, load_panel, scores_dir, split_path  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

OUT = os.path.join(ROOT, "results", "joint_removal_audit")

GAMMA = 0.20
TARGETS = ["J07", "J08", "J11"]
TARGET_NAMES = {
    "J07": "meta-llama/Llama-3.2-3B-Instruct",
    "J08": "meta-llama/Llama-3.1-8B-Instruct",
    "J11": "microsoft/Phi-4-mini-instruct",
}
PUBLISHED_TRIPLE_V = {20260818: 0.216, 20260819: 0.211, 20260820: 0.209,
                      20260821: 0.214, 20260822: 0.208}
PRIMARY_COUNTS = {"FIT": (1866, 935), "CERT": (932, 467), "TEST": (926, 463)}
BOOT_BASE_SEED = 20261108
PARTS = {"F": "FIT", "V": "CERT", "T": "TEST"}
EXPECTED_CONTEXTS = ["I0_clean", "I1_order_swap", "I2_correctness_rubric",
                     "I3_bias_resistant_rubric", "I4_verbosity_pad_a", "I5_source_tags",
                     "I6_reference_hidden"]

# Tolerances.  The repo fitter minimises max_c f_c + 1e-5 * mean_c f_c and
# accepts a stalled gap up to 1e-5, so its worst-context value can sit up to
# about 2e-5 above the true optimum.  Fit-only inequalities are checked at
# that scale; it is three orders of magnitude below the reported decimals.
FIT_TOL = 3e-5
LP_AGREE_TOL = 3e-5
IDENTITY_TOL = 1e-10
BOUND_TOL = 1e-10
BRACKET_SLACKS = [1e-6, 1e-4]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def label(R) -> str:
    return "+".join(R)


def run_split(seed: int, n_boot: int, do_bracket: bool, do_lp: bool, log) -> Dict[str, object]:
    panel = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=seed)
    ids = panel.judge_ids
    N = len(ids)
    # identity checks: exact names, not positions
    assert ids == sorted(CORE20), ids
    for j, name in zip(ids, panel.model_ids):
        assert name == CORE20[j], (j, name)
    for t in TARGETS:
        assert panel.model_ids[ids.index(t)] == TARGET_NAMES[t]
    idx = {t: ids.index(t) for t in TARGETS}

    P = {q: jr.dense(panel.splits[name]) for q, name in PARTS.items()}
    groups = {q: panel.item_groups[name] for q, name in PARTS.items()}
    checks = {
        "seed": seed,
        "shape": {q: list(P[q].shape) for q in P},
        "pairs": {q: int(P[q].shape[2]) for q in P},
        "tasks": {q: int(len(np.unique(groups[q]))) for q in P},
        "prob_min": float(min(P[q].min() for q in P)),
        "prob_row_sum_max_dev": float(max(np.abs(P[q].sum(-1) - 1).max() for q in P)),
        "contexts": panel.context_names,
    }
    assert sum(checks["pairs"].values()) == 3724
    assert sum(checks["tasks"].values()) == 1865
    assert all(np.isfinite(P[q]).all() for q in P)
    assert checks["prob_min"] >= -1e-6, checks["prob_min"]
    assert checks["prob_row_sum_max_dev"] <= 1e-4, checks["prob_row_sum_max_dev"]
    assert panel.context_names == EXPECTED_CONTEXTS, panel.context_names
    # no source task crosses partitions (groups are per-partition ids, so check on base items)
    with open(split_path(seed)) as f:
        assignment = json.load(f)["splits"]
    seen = [set(v) for v in assignment.values()]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    if seed == PRIMARY_SEED:
        for q, name in PARTS.items():
            assert (checks["pairs"][q], checks["tasks"][q]) == PRIMARY_COUNTS[name], (q, checks)

    # ---------------- fits (FIT only) ----------------
    sets = jr.removal_sets(TARGETS)
    weights: Dict[tuple, np.ndarray] = {}
    fit_records: List[dict] = []
    all_j = list(range(N))
    for R in sets:
        rem = [idx[t] for t in R]
        surv = [j for j in all_j if j not in rem]
        for t in R:
            i = idx[t]
            t0 = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                err, w = solve_minimax_weights(panel.splits["FIT"], i, surv)
            dt = time.time() - t0
            stalled = [str(c.message) for c in caught if "stalled" in str(c.message)]
            # the fitter raises unless it converges (gap <= 1e-7) or stalls within 1e-5
            status = "accepted_stall" if stalled else "converged"
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(P["F"][[i] + surv]).tobytes())
            row = np.zeros(N)
            row[surv] = w
            weights[(R, t)] = row
            rec = {
                "split_seed": seed, "removal_set": label(R), "target": t,
                "support_size": len(surv), "fit_objective_F": float(err),
                "fit_seconds": dt, "solver_status": status, "stall_messages": stalled,
                "fit_input_sha256": h.hexdigest(), "solver": "epcrc.judge.solve_minimax_weights (cutting planes, HiGHS master)",
                "provenance": "recomputed (E0 stored no weights)",
                "min_weight": float(w.min()), "weight_sum_err": float(abs(w.sum() - 1)),
                "weights": {ids[j]: float(row[j]) for j in all_j if row[j] > 0},
            }
            if do_lp:
                t0 = time.time()
                lp_err, _ = solve_minimax_weights_lp(panel.splits["FIT"], i, surv)
                rec["exact_lp_objective_F"] = float(lp_err)
                rec["exact_lp_seconds"] = time.time() - t0
                rec["cut_minus_lp"] = float(err - lp_err)
            fit_records.append(rec)
            log(f"  fit {seed} {label(R):<12} {t}: F={err:.6f} ({dt:.1f}s)"
                + (f" lp={rec['exact_lp_objective_F']:.6f}" if do_lp else ""))

    W_single = np.zeros((N, N))
    for t in TARGETS:
        W_single[idx[t]] = weights[((t,), t)]
    for (R, t), row in weights.items():
        assert all(row[idx[u]] == 0.0 for u in R), (R, t)   # no removed donor, W_ii = 0
        assert row.min() >= 0 and abs(row.sum() - 1) < 1e-9, (R, t)

    # ---------------- scoring ----------------
    losses: Dict[str, np.ndarray] = {}
    endpoint_rows, context_rows, coverage_rows = [], [], []
    for R in sets:
        rem = [idx[t] for t in R]
        W_R = np.stack([weights[(R, t)] for t in R])
        W_I = np.stack([weights[((t,), t)] for t in R])
        for q in P:
            L_R = jr.row_losses(P[q], rem, W_R)
            L_I = jr.row_losses(P[q], rem, W_I)
            losses[f"{seed}|{q}|{label(R)}|refit"] = L_R
            sR, sI = jr.summarise(L_R), jr.summarise(L_I)
            for k, t in enumerate(R):
                endpoint_rows.append({
                    "split_seed": seed, "partition": q, "removal_set": label(R), "target": t,
                    "individual_error": float(sI["endpoint_errors"][k]),
                    "refit_error": float(sR["endpoint_errors"][k]),
                    "delta": float(sR["endpoint_errors"][k] - sI["endpoint_errors"][k]),
                    "worst_context": panel.context_names[int(np.argmax(sR["context_means"][k]))],
                })
                for c, cname in enumerate(panel.context_names):
                    context_rows.append({
                        "split_seed": seed, "partition": q, "removal_set": label(R), "target": t,
                        "context": cname,
                        "individual_mean_tv": float(sI["context_means"][k, c]),
                        "refit_mean_tv": float(sR["context_means"][k, c]),
                    })
            coverage_rows.append({
                "split_seed": seed, "partition": q, "removal_set": label(R), "size": len(R),
                "E_individual": sI["coverage_error"], "E_refit": sR["coverage_error"],
                "Delta": sR["coverage_error"] - sI["coverage_error"],
                "margin": sR["coverage_error"] - GAMMA,
                "pass_at_gamma": bool(sR["coverage_error"] <= GAMMA),
                "worst_target": R[sR["worst_target"]],
                "worst_context": panel.context_names[sR["worst_context"]],
            })

    # ---------------- reproduction against archived E0 ----------------
    with open(os.path.join(ROOT, "results", "e0_real_core20.json")) as f:
        e0 = json.load(f)
    rec0 = [r for r in e0["records"] if r["seed"] == seed and abs(r["gamma"] - GAMMA) < 1e-12]
    assert len(rec0) == 1
    rec0 = rec0[0]
    trip = [r for r in coverage_rows if r["partition"] == "V" and r["removal_set"] == label(TARGETS)][0]
    single_v = {r["target"]: r["refit_error"] for r in endpoint_rows
                if r["partition"] == "V" and r["removal_set"] == r["target"]}
    repro = {
        "split_seed": seed,
        "triple_V_recomputed": trip["E_refit"],
        "triple_V_archived_full_precision": rec0["naive_coverage"],
        "triple_V_abs_diff_vs_archive": abs(trip["E_refit"] - rec0["naive_coverage"]),
        "triple_V_published_3dp": PUBLISHED_TRIPLE_V[seed],
        "triple_V_matches_published_rounding": abs(trip["E_refit"] - PUBLISHED_TRIPLE_V[seed]) <= 0.000500001,
        "archived_individually_removable": rec0["individually_removable"],
        "singleton_V_recomputed": single_v,
        "singleton_V_archived": {t: rec0["loo_errors"][idx[t]] for t in TARGETS},
        "singleton_all_pass": all(v <= GAMMA for v in single_v.values()),
        "removable_set_recomputed": None,
    }
    repro["singleton_max_abs_diff_vs_archive"] = max(
        abs(single_v[t] - repro["singleton_V_archived"][t]) for t in TARGETS)

    # ---------------- composition (Theorem 2) ----------------
    comp_rows, comp_ep_rows, edge_rows, matrices = [], [], [], {}
    numerical = []
    for R in sets:
        rem = [idx[t] for t in R]
        surv = [j for j in all_j if j not in rem]
        if len(R) == 1:
            continue
        A, B = jr.split_witness(W_single, rem, surv)
        for a, t in enumerate(R):
            for b, u in enumerate(R):
                if a != b:
                    edge_rows.append({"split_seed": seed, "removal_set": label(R),
                                      "target": t, "donor": u, "weight": float(A[a, b])})
        comp = jr.compose(A, B)
        matrices[f"{seed}|{label(R)}|A"] = A
        matrices[f"{seed}|{label(R)}|B"] = B
        base = {"split_seed": seed, "removal_set": label(R), "status": comp["status"],
                "rho_A": comp["rho"], "cond_I_minus_A": comp["condition_number"],
                "max_dependency_mass": float(comp["dependency_mass"].max()),
                "min_survivor_mass": float(comp["survivor_mass"].min())}
        if comp["status"] != "ok":
            for q in P:
                comp_rows.append(dict(base, partition=q))
            continue
        matrices[f"{seed}|{label(R)}|H"] = comp["H"]
        matrices[f"{seed}|{label(R)}|M"] = comp["M"]
        base.update(max_expected_depth=float(comp["expected_depth"].max()),
                    min_M=comp["min_M"], M_row_sum_error=comp["row_sum_error"])
        for q in P:
            d = jr.composition_on(P[q], rem, surv, W_single, comp)
            losses[f"{seed}|{q}|{label(R)}|comp"] = d["losses"]
            refit = [r for r in coverage_rows if r["partition"] == q and r["removal_set"] == label(R)][0]
            comp_rows.append(dict(
                base, partition=q,
                E_comp=d["coverage_error"], E_refit=refit["E_refit"],
                comp_minus_refit=d["coverage_error"] - refit["E_refit"],
                E_individual=refit["E_individual"],
                bound_raw=d["bound"], bound_clipped=d["bound_clipped"],
                bound_context=d["bound_context"],
                comp_pass_at_gamma=bool(d["coverage_error"] <= GAMMA),
                bound_certifies_gamma=bool(d["bound"] <= GAMMA),
            ))
            for k, t in enumerate(R):
                rf = [r for r in endpoint_rows if r["partition"] == q and r["removal_set"] == label(R)
                      and r["target"] == t][0]
                comp_ep_rows.append({
                    "split_seed": seed, "partition": q, "removal_set": label(R), "target": t,
                    "epsilon_singleton": float(d["epsilon"][k]),
                    "comp_error": float(d["row_errors"][k]),
                    "refit_error": rf["refit_error"],
                    "bound": float(d["row_bounds"][k]),
                    "bound_context": float(d["row_bounds_context"][k]),
                    "expected_depth": float(comp["expected_depth"][k]),
                    "survivor_mass": float(comp["survivor_mass"][k]),
                })
            numerical.append({
                "split_seed": seed, "removal_set": label(R), "partition": q,
                "identity_max_abs": d["identity_max_abs"],
                "identity_ok": d["identity_max_abs"] <= IDENTITY_TOL * max(1.0, comp["condition_number"]),
                "max_bound_excess": d["max_bound_excess"],
                "bound_ok": d["max_bound_excess"] <= BOUND_TOL,
                "max_context_bound_excess": d["max_context_bound_excess"],
                "context_bound_ok": d["max_context_bound_excess"] <= BOUND_TOL,
                "context_bound_le_bound": bool(np.all(d["row_bounds_context"] <= d["row_bounds"] + 1e-12)),
                "min_M": comp["min_M"], "M_row_sum_error": comp["row_sum_error"],
            })

    # fit-only invariants: support monotonicity and refit <= composition on F
    mono = []
    for R in sets:
        for R2 in sets:
            if set(R) < set(R2):
                for t in R:
                    e1 = [r for r in endpoint_rows if r["partition"] == "F" and r["removal_set"] == label(R) and r["target"] == t][0]["refit_error"]
                    e2 = [r for r in endpoint_rows if r["partition"] == "F" and r["removal_set"] == label(R2) and r["target"] == t][0]["refit_error"]
                    mono.append({"split_seed": seed, "smaller": label(R), "larger": label(R2), "target": t,
                                 "e_smaller_removal": e1, "e_larger_removal": e2,
                                 "ok": e2 >= e1 - FIT_TOL})
    refit_le_comp = []
    for r in comp_ep_rows:
        if r["partition"] == "F":
            refit_le_comp.append({"split_seed": seed, "removal_set": r["removal_set"], "target": r["target"],
                                  "refit_minus_comp": r["refit_error"] - r["comp_error"],
                                  "ok": r["refit_error"] <= r["comp_error"] + FIT_TOL})

    # ---------------- bootstrap ----------------
    # Fixed coordinate for the smooth check: the (target, context) that is
    # worst for the triple refit on V.  Fixed from V, then tested on T.
    trip_key = label(TARGETS)
    rem3 = [idx[t] for t in TARGETS]
    Vmeans = losses[f"{seed}|V|{trip_key}|refit"].mean(-1)
    fixed_t, fixed_c = np.unravel_index(int(np.argmax(Vmeans)), Vmeans.shape)
    coord_rows = []
    for q in ("V", "T"):
        W_I = np.stack([weights[((t,), t)] for t in TARGETS])
        L_I = jr.row_losses(P[q], rem3, W_I)
        L_R = losses[f"{seed}|{q}|{trip_key}|refit"]
        s = jr.stable_seed(BOOT_BASE_SEED, seed, q, trip_key, "coordinates")
        cd = jr.coordinate_deltas(L_I, L_R, groups[q], n_boot, s)
        for k, t in enumerate(TARGETS):
            for c, cname in enumerate(panel.context_names):
                coord_rows.append({
                    "split_seed": seed, "partition": q, "removal_set": trip_key,
                    "target": t, "context": cname,
                    "delta": float(cd["point"][k, c]), "sd": float(cd["sd"][k, c]),
                    "ci_low": float(cd["ci_low"][k, c]), "ci_high": float(cd["ci_high"][k, c]),
                    "maxt_band_low": float(cd["band_low"][k, c]),
                    "maxt_band_high": float(cd["band_high"][k, c]),
                    "maxt_q": cd["maxt_q"],
                    "fixed_coordinate_from_V": bool(k == fixed_t and c == fixed_c),
                })

    boot_rows = []
    for q in ("V", "T"):
        for R in sets:
            if len(R) == 1:
                continue
            rem = [idx[t] for t in R]
            W_I = np.stack([weights[((t,), t)] for t in R])
            L_I = jr.row_losses(P[q], rem, W_I)
            L_R = losses[f"{seed}|{q}|{label(R)}|refit"]
            fam = 5 if len(R) == 3 else 1
            s = jr.stable_seed(BOOT_BASE_SEED, seed, q, label(R), "ind_vs_refit")
            for row in jr.paired_comparison(L_I, L_R, groups[q], GAMMA, n_boot, s, family_size=fam):
                row.update(split_seed=seed, partition=q, removal_set=label(R),
                           comparison="individual_vs_refit",
                           target=("ALL" if row["target"] < 0 else R[row["target"]]),
                           role=(("inferential_check" if q == "T" else "descriptive_V_selected")
                                 if len(R) == 3 else "exploratory"))
                boot_rows.append(row)
            key = f"{seed}|{q}|{label(R)}|comp"
            if key in losses:
                s = jr.stable_seed(BOOT_BASE_SEED, seed, q, label(R), "refit_vs_comp")
                for row in jr.paired_comparison(L_R, losses[key], groups[q], GAMMA, n_boot, s):
                    row.update(split_seed=seed, partition=q, removal_set=label(R),
                               comparison="refit_vs_comp",
                               target=("ALL" if row["target"] < 0 else R[row["target"]]),
                               role="supporting")
                    boot_rows.append(row)

    # ---------------- near-optimal witness bracket ----------------
    # Per removal set R and target i, bracket the singleton row's mass on
    # R \ {i} (the entries that become A for this R) over all witnesses whose
    # F objective is within `slack` of the exact optimum.  This is an exact LP
    # per row.  It brackets row masses only; rho, H, the bound and E_comp are
    # nonlinear in W, so they are reported at the two extreme witnesses as
    # illustrations, not as bounds.
    bracket_rows = []
    if do_bracket:
        opts = {}
        for t in TARGETS:
            i = idx[t]
            known = [r.get("exact_lp_objective_F") for r in fit_records
                     if r["removal_set"] == t and r["target"] == t]
            if known and known[0] is not None:
                opts[t] = known[0]
            else:
                opts[t], _ = solve_minimax_weights_lp(panel.splits["FIT"], i, [j for j in all_j if j != i])
        for slack in BRACKET_SLACKS:
            for R in sets:
                if len(R) == 1:
                    continue
                rem = [idx[t] for t in R]
                surv = [j for j in all_j if j not in rem]
                ends = {"min": np.zeros((N, N)), "max": np.zeros((N, N))}
                for t in R:
                    i = idx[t]
                    donors = [j for j in all_j if j != i]
                    others = [idx[u] for u in R if u != t]
                    br = jr.bracket_mass(P["F"], i, donors, others, opts[t], slack)
                    orig = float(sum(W_single[i, j] for j in others))
                    for sense in ("min", "max"):
                        ends[sense][i] = br[sense]["row"]
                    bracket_rows.append({
                        "split_seed": seed, "slack": slack, "removal_set": label(R),
                        "kind": "row_mass", "target": t, "exact_opt_F": opts[t],
                        "mass_original": orig,
                        "mass_min": br["min"]["mass"], "mass_max": br["max"]["mass"],
                        "fit_err_at_min": br["min"]["fit_error"], "fit_err_at_max": br["max"]["fit_error"],
                    })
                    log(f"  bracket {seed} slack={slack:g} {label(R)} {t}: orig {orig:.4f} "
                        f"range [{br['min']['mass']:.4f}, {br['max']['mass']:.4f}]")
                for sense in ("min", "max"):
                    A, B = jr.split_witness(ends[sense], rem, surv)
                    comp = jr.compose(A, B)
                    row = {"split_seed": seed, "slack": slack, "removal_set": label(R),
                           "kind": f"composition_at_{sense}_mass_witness",
                           "status": comp["status"], "rho_A": comp["rho"]}
                    if comp["status"] == "ok":
                        for q in P:
                            d = jr.composition_on(P[q], rem, surv, ends[sense], comp)
                            row[f"E_comp_{q}"] = d["coverage_error"]
                            row[f"bound_{q}"] = d["bound"]
                    bracket_rows.append(row)

    return {
        "checks": checks, "fit_records": fit_records, "weights": weights,
        "losses": losses, "endpoint": endpoint_rows, "context": context_rows,
        "coverage": coverage_rows, "repro": repro, "composition": comp_rows,
        "composition_endpoint": comp_ep_rows, "edges": edge_rows, "matrices": matrices,
        "numerical": numerical, "monotonicity": mono, "refit_le_comp": refit_le_comp,
        "bootstrap": boot_rows, "coordinates": coord_rows,
        "bracket": bracket_rows, "judge_ids": ids,
        "contexts": panel.context_names,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="*", default=SPLIT_SEEDS)
    ap.add_argument("--bracket-seeds", type=int, nargs="*", default=[PRIMARY_SEED],
                    help="splits that get the near-optimal witness bracket (exact LPs, slow)")
    ap.add_argument("--lp-seeds", type=int, nargs="*", default=[PRIMARY_SEED],
                    help="splits whose fits are cross-checked against the exact monolithic LP (slow)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "run.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    results = []
    for seed in args.seeds:
        log(f"[split {seed}]")
        results.append(run_split(seed, args.boot, seed in args.bracket_seeds, seed in args.lp_seeds, log))

    def cat(key):
        return pd.DataFrame([r for res in results for r in res[key]])

    tables = {
        "endpoint_errors.csv": cat("endpoint"), "context_errors.csv": cat("context"),
        "coverage_errors.csv": cat("coverage"), "composition_metrics.csv": cat("composition"),
        "composition_endpoint_metrics.csv": cat("composition_endpoint"),
        "dependency_edges.csv": cat("edges"), "bootstrap_statistics.csv": cat("bootstrap"),
        "coordinate_deltas.csv": cat("coordinates"),
    }
    if any(res["bracket"] for res in results):
        tables["witness_bracket.csv"] = cat("bracket")
    for name, df in tables.items():
        df.to_csv(os.path.join(args.out, name), index=False, float_format="%.10g")

    np.savez_compressed(os.path.join(args.out, "losses.npz"),
                        **{k: v for res in results for k, v in res["losses"].items()})
    np.savez_compressed(os.path.join(args.out, "matrices.npz"),
                        **{k: v for res in results for k, v in res["matrices"].items()})
    np.savez_compressed(os.path.join(args.out, "weights.npz"),
                        **{f"{res['checks']['seed']}|{label(R)}|{t}": w
                           for res in results for (R, t), w in res["weights"].items()},
                        judge_ids=np.array(results[0]["judge_ids"]))

    with open(os.path.join(args.out, "fit_records.json"), "w") as f:
        json.dump([r for res in results for r in res["fit_records"]], f, indent=1)
    e1 = json.load(open(os.path.join(ROOT, "results", "e1_frontier_core20.json")))
    k17 = e1["methods"]["coverage_backward"]["chain"]["17"]
    t_exposure = {
        "e1_backward_k17_removed": sorted(set(CORE20) - set(k17)),
        "e1_backward_k17_is_this_triple": sorted(set(CORE20) - set(k17)) == TARGETS,
        "e1_primary_split_TEST_k17": e1["methods"]["coverage_backward"]["TEST"]["17"]["worst_judge_worst_context_tv"],
        "this_run_primary_TEST_triple": [r for res in results for r in res["coverage"]
                                         if r["split_seed"] == PRIMARY_SEED and r["partition"] == "T"
                                         and r["removal_set"] == label(TARGETS)][0]["E_refit"]
        if PRIMARY_SEED in args.seeds else None,
        "note": "TEST was already reported for this exact deletion (backward elimination, k=17) in the paper's held-out frontier; the T check is fixed-design, not untouched data.",
    }
    with open(os.path.join(args.out, "test_exposure.json"), "w") as f:
        json.dump(t_exposure, f, indent=1)
    inventory = {
        "response_cache": os.path.relpath(scores_dir("core20"), ROOT) + "/J??__I?_*.json (140 blocks: 20 judges x 7 contexts)",
        "cache_loader": "epcrc/panel.py: load_panel / load_raw_panel / split_panel",
        "pairs_file": f"data/pairs_seed{PRIMARY_SEED}.jsonl (fixes which 3724 pairs and their text)",
        "split_manifests": [os.path.relpath(split_path(s), ROOT) for s in SPLIT_SEEDS],
        "source_task_ids": "Pair.base_item_id -> Panel.item_groups per partition",
        "independent_removal_experiment": "experiments/experiment_e0_noncomposability.py (results/e0_real_core20.json)",
        "minimax_tv_solver": "epcrc/judge.py: solve_minimax_weights (cutting planes, used for all fits); solve_minimax_weights_lp (exact LP, cross-check)",
        "archived_weights": "none: E0 stored errors and LOO weights only implicitly; all 60 fits recomputed with the original solver and validated against archived full-precision errors",
        "archived_full_precision_errors": "results/e0_real_core20.json (loo_errors, naive_coverage at gamma=0.20)",
        "exact_F_minimum": "results/backbone_core20.json (k*=18 at gamma=0.20, certified, all 5 splits)",
        "plotting": "experiments/report_e9_joint_removal.py",
        "repository_revision": None,
        "model_revisions": "the score blocks store revision=None (hub default at scoring time); the manuscript's Appendix B Table 2 lists revision prefixes, e.g. J07 0cb88a4f764b, J08 0e9e39f249a1, J11 cfbefacb9925 (protocol 1.2)",
    }
    with open(os.path.join(args.out, "replication_check.json"), "w") as f:
        json.dump([res["repro"] for res in results], f, indent=1)

    numerical = {
        "data_checks": [res["checks"] for res in results],
        "composition": [r for res in results for r in res["numerical"]],
        "support_monotonicity_F": [r for res in results for r in res["monotonicity"]],
        "refit_le_composition_F": [r for res in results for r in res["refit_le_comp"]],
        "fit_lp_agreement": [
            {k: r[k] for k in ("split_seed", "removal_set", "target", "cut_minus_lp")}
            for res in results for r in res["fit_records"] if "cut_minus_lp" in r],
        "tolerances": {"FIT_TOL": FIT_TOL, "LP_AGREE_TOL": LP_AGREE_TOL,
                       "IDENTITY_TOL": IDENTITY_TOL, "BOUND_TOL": BOUND_TOL},
    }
    summary = {
        "all_identity_ok": all(r["identity_ok"] for r in numerical["composition"]),
        "all_bound_ok": all(r["bound_ok"] for r in numerical["composition"]),
        "all_context_bound_ok": all(r["context_bound_ok"] for r in numerical["composition"]),
        "all_monotone_F": all(r["ok"] for r in numerical["support_monotonicity_F"]),
        "all_refit_le_comp_F": all(r["ok"] for r in numerical["refit_le_composition_F"]),
        "all_fit_lp_agree": all(r["cut_minus_lp"] >= -LP_AGREE_TOL and r["cut_minus_lp"] <= LP_AGREE_TOL
                                for r in numerical["fit_lp_agreement"]) if numerical["fit_lp_agreement"] else None,
        "max_abs_cut_minus_lp": max((abs(r["cut_minus_lp"]) for r in numerical["fit_lp_agreement"]), default=None),
        "all_repro_match_archive": all(res["repro"]["triple_V_abs_diff_vs_archive"] < 1e-5
                                       and res["repro"]["singleton_max_abs_diff_vs_archive"] < 1e-5
                                       for res in results),
        "all_repro_match_published": all(res["repro"]["triple_V_matches_published_rounding"] for res in results),
        "all_singletons_pass": all(res["repro"]["singleton_all_pass"] for res in results),
    }
    numerical["summary"] = summary
    with open(os.path.join(args.out, "numerical_checks.json"), "w") as f:
        json.dump(numerical, f, indent=1, default=float)

    inputs = [os.path.join(scores_dir("core20"), p) for p in sorted(os.listdir(scores_dir("core20")))
              if p.startswith("J")]
    inputs += [split_path(s) for s in args.seeds]
    inputs += [os.path.join(ROOT, "data", f"pairs_seed{PRIMARY_SEED}.jsonl"),
               os.path.join(ROOT, "results", "e0_real_core20.json")]
    try:
        rev = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"], text=True).strip())
    except Exception:
        rev, dirty = "unknown", None
    manifest = {
        "experiment": "E9 joint removal audit (protocol 1.0, 2026-11-08)",
        "command": "python -u " + " ".join(sys.argv),
        "git_revision": rev, "git_dirty": dirty,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "platform": platform.platform(),
        "gamma": GAMMA, "targets": TARGET_NAMES, "split_seeds": args.seeds,
        "bootstrap_resamples": args.boot, "bootstrap_base_seed": BOOT_BASE_SEED,
        "bracket_seeds": args.bracket_seeds, "bracket_slacks": BRACKET_SLACKS,
        "exact_lp_check_seeds": args.lp_seeds, "elapsed_seconds": time.time() - t0,
        "inputs_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in inputs},
        "contexts": results[0]["contexts"],
        "summary_checks": summary,
        "solver_status_counts": pd.Series([r["solver_status"] for res in results for r in res["fit_records"]]).value_counts().to_dict(),
    }
    inventory["repository_revision"] = rev
    with open(os.path.join(args.out, "data_inventory.json"), "w") as f:
        json.dump(inventory, f, indent=1)
    with open(os.path.join(args.out, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out, "RUN_COMMAND.txt"), "w") as f:
        f.write(manifest["command"] + "\n")
    log(json.dumps(summary, indent=1))
    log(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
