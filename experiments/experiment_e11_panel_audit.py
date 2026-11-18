"""E11: the individual-versus-joint removal audit, for any registered panel.

E9 audited one removal set on Core-20.  This runs the same audit without
naming the set in advance, so it can be repeated on a second panel and on the
union of the two.

Per split seed, at one predeclared tolerance gamma (default 0.20):

  1. Fit every judge against all the others on F.  Record its error (worst
     context of the mean TV) on F, V and T.
  2. R = the judges whose individual error on V is at most gamma.
  3. Remove all of R at once: refit each member against the survivors on F and
     record the joint error.  Delta = joint error - largest individual error.
  4. Every pair and triple inside R (when there are at most `--max-subsets` of
     them): joint error on V, and the minimal subsets that fail.
  5. A jointly removable subset built greedily: candidates in order of
     individual V error, each kept if the joint V error stays within gamma.
  6. Theorem 2: compose the individual maps, M = (I - A)^-1 B, and compare the
     composed map and its bound with the refit.
  7. Paired source-task bootstrap of Delta on V and T.

`--focus` reports a named set as well (its individual and joint error in this
panel), which is how the Core-20 removal set is followed into the union.

Usage:
    python -u experiments/experiment_e11_panel_audit.py --panel core20
Writes results/e11_panel_audit_<panel>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from itertools import combinations
from math import comb
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.judge import solve_minimax_weights  # noqa: E402
from epcrc.panel import PANELS, SPLIT_SEEDS, load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

PARTS = {"F": "FIT", "V": "CERT", "T": "TEST"}
BOOT_BASE_SEED = 20261120


def fit_rows(fit, targets, support, N):
    """Weight rows [len(targets), N] of each target fitted on `support`."""
    W = np.zeros((len(targets), N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r, i in enumerate(targets):
            _, w = solve_minimax_weights(fit, i, list(support))
            W[r, list(support)] = w
    return W


def errors(P, targets, W):
    """Per-target error on each partition: {q: [len(targets)]}."""
    return {q: jr.row_losses(P[q], targets, W).mean(-1).max(-1) for q in P}


def audit_set(panel, P, removed, W_single, groups, gamma, n_boot, seed, tag):
    """Joint removal of `removed`: refit, Delta, Theorem 2 and the bootstrap."""
    ids, N = panel.judge_ids, panel.N
    removed = sorted(removed)
    surv = [j for j in range(N) if j not in removed]
    W_I = W_single[removed]
    W_R = fit_rows(panel.splits["FIT"], removed, surv, N)
    eI, eR = errors(P, removed, W_I), errors(P, removed, W_R)
    out = {
        "removed": [ids[i] for i in removed],
        "individual": {q: {ids[i]: float(v) for i, v in zip(removed, eI[q])} for q in P},
        "joint": {q: {ids[i]: float(v) for i, v in zip(removed, eR[q])} for q in P},
        "individual_max": {q: float(eI[q].max()) for q in P},
        "joint_max": {q: float(eR[q].max()) for q in P},
        "delta": {q: float(eR[q].max() - eI[q].max()) for q in P},
        "joint_within_gamma": {q: bool(eR[q].max() <= gamma) for q in P},
    }
    for q in P:
        s = jr.summarise(jr.row_losses(P[q], removed, W_R))
        out.setdefault("worst", {})[q] = {"target": ids[removed[s["worst_target"]]],
                                           "context": panel.context_names[s["worst_context"]]}
    if len(removed) >= 2:
        A, B = jr.split_witness(W_single, removed, surv)
        comp = jr.compose(A, B)
        th = {"status": comp["status"], "rho": comp["rho"],
              "dependency_mass": {ids[i]: float(v) for i, v in zip(removed, comp["dependency_mass"])}}
        if comp["status"] == "ok":
            for q in P:
                d = jr.composition_on(P[q], removed, surv, W_single, comp)
                th[q] = {"composed_error": d["coverage_error"], "bound": d["bound"],
                         "refit_error": out["joint_max"][q],
                         "refit_le_composed": bool(out["joint_max"][q] <= d["coverage_error"] + 1e-6 or q != "F"),
                         "max_bound_excess": d["max_bound_excess"]}
        out["theorem2"] = th
        out["bootstrap"] = {}
        for q in ("V", "T"):
            rows = jr.paired_comparison(jr.row_losses(P[q], removed, W_I), jr.row_losses(P[q], removed, W_R),
                                        groups[q], gamma, n_boot, jr.stable_seed(BOOT_BASE_SEED, seed, q, tag))
            agg = rows[0]
            out["bootstrap"][q] = {k: agg[k] for k in (
                "delta", "delta_ci_low", "delta_ci_high", "frac_draws_delta_le_0",
                "margin", "margin_ci_low", "margin_ci_high", "boot_bias_other", "n_boot")}
    return out


def run_split(args):
    panel_name, seed, gamma, n_boot, max_subsets, focus = args
    t0 = time.time()
    panel = load_panel(PRIMARY_SEED, scores_dir(panel_name), split_seed=seed)
    ids, N = panel.judge_ids, panel.N
    fit = panel.splits["FIT"]
    P = {q: jr.dense(panel.splits[name]) for q, name in PARTS.items()}
    groups = {q: panel.item_groups[name] for q, name in PARTS.items()}

    W_single = np.vstack([fit_rows(fit, [i], [j for j in range(N) if j != i], N) for i in range(N)])
    ind = errors(P, range(N), W_single)
    R = [i for i in range(N) if ind["V"][i] <= gamma]
    out = {
        "seed": seed, "judges": ids, "n_pairs": {q: int(P[q].shape[2]) for q in P},
        "individual": {q: {ids[i]: float(ind[q][i]) for i in range(N)} for q in P},
        "removable_on_V": [ids[i] for i in R],
    }

    def joint_V(sub):
        sub = sorted(sub)
        W = fit_rows(fit, sub, [j for j in range(N) if j not in sub], N)
        return float(jr.row_losses(P["V"], sub, W).mean(-1).max())

    if len(R) >= 2:
        out["all_removable"] = audit_set(panel, P, R, W_single, groups, gamma, n_boot, seed, "R")
        n_sub = sum(comb(len(R), k) for k in (2, 3) if k < len(R))
        if 0 < n_sub <= max_subsets:
            sub_err = {c: joint_V(c) for k in (2, 3) if k < len(R) for c in combinations(R, k)}
            failing = {c for c, e in sub_err.items() if e > gamma}
            minimal = [c for c in failing if not any(set(d) < set(c) for d in failing)]
            out["subsets"] = {
                "n_enumerated": len(sub_err), "n_failing_on_V": len(failing),
                "minimal_failing": [{"set": [ids[i] for i in c], "joint_V": sub_err[c]} for c in sorted(minimal)],
                "errors": [{"set": [ids[i] for i in c], "joint_V": e} for c, e in sorted(sub_err.items())],
            }
        else:
            out["subsets"] = {"skipped": f"{n_sub} pairs and triples"}
        kept, trail = [], []
        for c in sorted(R, key=lambda i: ind["V"][i]):
            e = joint_V(kept + [c]) if kept else float(ind["V"][c])
            ok = e <= gamma
            trail.append({"candidate": ids[c], "joint_V_if_added": e, "kept": bool(ok)})
            if ok:
                kept.append(c)
        out["greedy_jointly_removable"] = {"set": [ids[i] for i in sorted(kept)], "size": len(kept), "trail": trail}

    if focus and all(f in ids for f in focus) and len(focus) >= 2:
        out["focus"] = audit_set(panel, P, [ids.index(f) for f in focus], W_single, groups,
                                 gamma, n_boot, seed, "focus")
    out["elapsed_seconds"] = time.time() - t0
    a = out.get("all_removable", {})
    print(seed, "R =", out["removable_on_V"], "| individual max", a.get("individual_max"),
          "| joint", a.get("joint_max"), f"| {out['elapsed_seconds']:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", choices=sorted(PANELS), required=True)
    ap.add_argument("--gamma", type=float, default=0.20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--max-subsets", type=int, default=400)
    ap.add_argument("--focus", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    jobs = [(a.panel, s, a.gamma, a.boot, a.max_subsets, a.focus) for s in a.seeds]
    with Pool(min(a.workers, len(jobs))) as pool:
        results = pool.map(run_split, jobs)
    path = os.path.join(ROOT, "results", f"e11_panel_audit_{a.panel}.json")
    with open(path, "w") as handle:
        json.dump({"panel": a.panel, "gamma": a.gamma, "n_boot": a.boot, "focus": a.focus,
                   "design": __doc__, "splits": results}, handle, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
