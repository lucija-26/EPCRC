"""E9 exact-LP checks, run in parallel after experiment_e9_joint_removal.py.

Two confirmations that change no reported number:

  1. every primary-split fit's F objective (cutting planes, the repo's fitter)
     is compared with the exact monolithic LP (`solve_minimax_weights_lp`);
  2. for every pair/triple R and target i in R, the singleton witness's row
     mass on R \\ {i} is bracketed over all F-witnesses within `slack` of the
     exact optimum (protocol section 3.8), and the composition is rerun at the
     two extreme witnesses.

Reads results/joint_removal_audit/weights.npz; writes exact_lp_check.json,
witness_bracket.csv and exact_checks_manifest.json next to it.

    python -u experiments/experiment_e9_exact_checks.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.judge import solve_minimax_weights_lp, worst_context_error  # noqa: E402
from epcrc.panel import load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

RES = os.path.join(ROOT, "results", "joint_removal_audit")
TARGETS = ["J07", "J08", "J11"]
SLACKS = [1e-6, 1e-4]
LP_AGREE_TOL = 3e-5

_P = None


def _init(seed):
    global _P
    panel = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=seed)
    _P = {"panel": panel, "F": jr.dense(panel.splits["FIT"]),
          "Q": {q: jr.dense(panel.splits[n]) for q, n in (("F", "FIT"), ("V", "CERT"), ("T", "TEST"))}}


def _lp(job):
    i, surv = job
    t0 = time.time()
    err, w = solve_minimax_weights_lp(_P["panel"].splits["FIT"], i, surv)
    return err, time.time() - t0


def _bracket(job):
    i, donors, others, opt, slack, sense = job
    t0 = time.time()
    br = jr.bracket_mass(_P["F"], i, donors, others, opt, slack, senses=(sense,))
    return br[sense], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-bracket", action="store_true", help="only the exact-LP agreement check")
    args = ap.parse_args()
    t0 = time.time()
    _init(args.seed)
    ids = _P["panel"].judge_ids
    idx = {t: ids.index(t) for t in TARGETS}
    N = len(ids)
    all_j = list(range(N))
    wz = np.load(os.path.join(RES, "weights.npz"))
    fits = json.load(open(os.path.join(RES, "fit_records.json")))
    fits = [r for r in fits if r["split_seed"] == args.seed]
    sets = jr.removal_sets(TARGETS)

    # ---- phase 1: exact LP for all 12 fits ----
    jobs, keys = [], []
    for R in sets:
        rem = [idx[t] for t in R]
        surv = [j for j in all_j if j not in rem]
        for t in R:
            jobs.append((idx[t], surv))
            keys.append(("+".join(R), t))
    with Pool(args.workers, initializer=_init, initargs=(args.seed,)) as pool:
        lp = pool.map(_lp, jobs, chunksize=1)
    lp_rows = []
    for (rs, t), (err, dt) in zip(keys, lp):
        cut = [r for r in fits if r["removal_set"] == rs and r["target"] == t][0]["fit_objective_F"]
        lp_rows.append({"split_seed": args.seed, "removal_set": rs, "target": t,
                        "cut_objective_F": cut, "exact_lp_objective_F": err,
                        "cut_minus_lp": cut - err, "lp_seconds": dt,
                        "agree": abs(cut - err) <= LP_AGREE_TOL})
        print(f"  lp {rs:<12} {t}: cut {cut:.7f} lp {err:.7f} diff {cut - err:+.2e} ({dt:.0f}s)", flush=True)
    lp_doc = {"rows": lp_rows, "all_agree": all(r["agree"] for r in lp_rows),
              "max_abs_diff": max(abs(r["cut_minus_lp"]) for r in lp_rows), "tolerance": LP_AGREE_TOL}
    with open(os.path.join(RES, f"exact_lp_check_{args.seed}.json"), "w") as f:
        json.dump(lp_doc, f, indent=1)
    if args.no_bracket:
        print(f"lp-only done in {time.time() - t0:.0f}s")
        return
    opts = {t: [r for r in lp_rows if r["removal_set"] == t and r["target"] == t][0]["exact_lp_objective_F"]
            for t in TARGETS}

    # ---- phase 2: bracket LPs ----
    W_single = np.zeros((N, N))
    for t in TARGETS:
        W_single[idx[t]] = wz[f"{args.seed}|{t}|{t}"]
    bjobs, bkeys = [], []
    for slack in SLACKS:
        for R in sets:
            if len(R) == 1:
                continue
            for t in R:
                i = idx[t]
                donors = [j for j in all_j if j != i]
                others = [idx[u] for u in R if u != t]
                for sense in ("min", "max"):
                    bjobs.append((i, donors, others, opts[t], slack, sense))
                    bkeys.append((slack, R, t, sense))
    with Pool(args.workers, initializer=_init, initargs=(args.seed,)) as pool:
        bres = pool.map(_bracket, bjobs, chunksize=1)
    got = {k: v for k, v in zip(bkeys, bres)}
    rows = []
    for slack in SLACKS:
        for R in sets:
            if len(R) == 1:
                continue
            rem = [idx[t] for t in R]
            surv = [j for j in all_j if j not in rem]
            ends = {"min": np.zeros((N, N)), "max": np.zeros((N, N))}
            for t in R:
                i = idx[t]
                others = [idx[u] for u in R if u != t]
                lo, hi = got[(slack, R, t, "min")][0], got[(slack, R, t, "max")][0]
                ends["min"][i], ends["max"][i] = lo["row"], hi["row"]
                rows.append({"split_seed": args.seed, "slack": slack, "removal_set": "+".join(R),
                             "kind": "row_mass", "target": t, "exact_opt_F": opts[t],
                             "mass_original": float(W_single[i, others].sum()),
                             "mass_min": lo["mass"], "mass_max": hi["mass"],
                             "fit_err_at_min": lo["fit_error"], "fit_err_at_max": hi["fit_error"]})
            for sense in ("min", "max"):
                A, B = jr.split_witness(ends[sense], rem, surv)
                comp = jr.compose(A, B)
                row = {"split_seed": args.seed, "slack": slack, "removal_set": "+".join(R),
                       "kind": f"composition_at_{sense}_mass_witness", "status": comp["status"],
                       "rho_A": comp["rho"]}
                if comp["status"] == "ok":
                    for q, Pq in _P["Q"].items():
                        d = jr.composition_on(Pq, rem, surv, ends[sense], comp)
                        row[f"E_comp_{q}"] = d["coverage_error"]
                        row[f"bound_{q}"] = d["bound"]
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "witness_bracket.csv"), index=False, float_format="%.10g")
    with open(os.path.join(RES, "exact_lp_check.json"), "w") as f:
        json.dump({"rows": lp_rows, "all_agree": all(r["agree"] for r in lp_rows),
                   "max_abs_diff": max(abs(r["cut_minus_lp"]) for r in lp_rows),
                   "tolerance": LP_AGREE_TOL}, f, indent=1)
    with open(os.path.join(RES, "exact_checks_manifest.json"), "w") as f:
        json.dump({"command": "python -u " + " ".join(sys.argv), "seed": args.seed, "slacks": SLACKS,
                   "workers": args.workers, "elapsed_seconds": time.time() - t0}, f, indent=1)
    print(pd.DataFrame([r for r in rows if r["kind"] == "row_mass"]).round(4).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
