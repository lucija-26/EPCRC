"""E9 rebuttal check -- refit-inclusive paired source-task bootstrap.

Reviewer items 3 and 6: the triple-removal margin on FIT (+0.0022..+0.0064 above
gamma = 0.20) is exact only for the fixed FIT sample, and the existing E9
bootstrap holds the fitted weights fixed, so it ignores fitting variability.

Design (fixed before running, not tuned).  For each split seed and each of B
draws, from one deterministic stream ``numpy.random.default_rng(20261110 + seed)``
drawn in the order FIT, CERT, TEST:

  1. resample FIT source tasks with replacement (every pair of a drawn task, with
     multiplicity) and build the resampled JudgeResponses by row indexing;
  2. refit (a) each singleton map J \\ {i}, i in {J07, J08, J11}, and (b) each
     triple-removal target on J \\ {J07, J08, J11}, on the resampled FIT with
     epcrc.judge.solve_minimax_weights;
  3. record on the resampled FIT: triple coverage error (max over targets of the
     worst context) and the max singleton error;
  4. score the refitted maps on CERT (V) and TEST (T) twice:
       * "fixed-eval":    on the fixed CERT / TEST partitions (fitting variability only);
       * "resampled-eval": on an independent source-task resample of CERT and of
         TEST drawn in the same draw (fitting + evaluation variability).

Reported per split: point estimates from the original fit, percentile 95%
intervals (and, as a secondary bias-aware column, basic/reverse-percentile
intervals 2*point - q), the fraction of draws with F triple error > gamma, and
the bias of the resampled-F triple error (mean draw - point).

Usage:
    .venv/bin/python -u experiments/e9_refit_bootstrap.py --boot 200 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.judge import JudgeResponses, solve_minimax_weights  # noqa: E402
from epcrc.panel import load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

OUT = os.path.join(ROOT, "results", "joint_removal_audit")
GAMMA = 0.20
TARGETS = ["J07", "J08", "J11"]
SEEDS = [20260818, 20260819, 20260820, 20260821, 20260822]
RNG_BASE = 20261110

_CACHE: dict = {}


def _panel(seed: int):
    if seed not in _CACHE:
        panel = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=seed)
        ids = panel.judge_ids
        idx = [ids.index(t) for t in TARGETS]
        N = len(ids)
        surv = [j for j in range(N) if j not in idx]
        _CACHE[seed] = {
            "panel": panel,
            "idx": idx,
            "N": N,
            "surv": surv,
            "P": {q: jr.dense(panel.splits[name]) for q, name in (("V", "CERT"), ("T", "TEST"))},
            "groups": {q: panel.item_groups[name] for q, name in
                       (("F", "FIT"), ("V", "CERT"), ("T", "TEST"))},
        }
    return _CACHE[seed]


def task_rows(groups: np.ndarray):
    """List of row-index arrays, one per source task (tasks are 0..G-1)."""
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    return np.split(order, bounds)


def fit_all(resp: JudgeResponses, d: dict):
    """Fit singletons and triple targets on `resp`.  Returns weight rows [N] and F errors."""
    N, idx, surv = d["N"], d["idx"], d["surv"]
    W_single, W_triple, e_single, e_triple = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in idx:
            kept = [j for j in range(N) if j != i]
            err, w = solve_minimax_weights(resp, i, kept)
            row = np.zeros(N)
            row[kept] = w
            W_single.append(row)
            e_single.append(err)
            err, w = solve_minimax_weights(resp, i, surv)
            row = np.zeros(N)
            row[surv] = w
            W_triple.append(row)
            e_triple.append(err)
    return (np.stack(W_single), np.stack(W_triple),
            float(max(e_single)), float(max(e_triple)))


def coverage(P: np.ndarray, idx, W: np.ndarray, row_weight=None) -> float:
    """max over targets and contexts of the (row-weighted) mean TV."""
    L = jr.row_losses(P, idx, W)           # [T, C, n]
    if row_weight is None:
        means = L.mean(axis=-1)
    else:
        means = (L * row_weight).sum(axis=-1) / row_weight.sum()
    return float(means.max())


def evaluate(d: dict, Ws, Wt, mult=None):
    out = {}
    for q in ("V", "T"):
        rw = None if mult is None else mult[q]
        out[q] = (coverage(d["P"][q], d["idx"], Wt, rw), coverage(d["P"][q], d["idx"], Ws, rw))
    return out


def point(seed: int):
    d = _panel(seed)
    Ws, Wt, es, et = fit_all(d["panel"].splits["FIT"], d)
    ev = evaluate(d, Ws, Wt)
    return {"F_triple": et, "F_single": es,
            "V_triple": ev["V"][0], "V_single": ev["V"][1],
            "T_triple": ev["T"][0], "T_single": ev["T"][1]}


def draw(args):
    seed, b, fit_rows, mult = args
    d = _panel(seed)
    fit = d["panel"].splits["FIT"]
    resp = JudgeResponses([blk[fit_rows] for blk in fit.blocks], fit.context_names)
    Ws, Wt, es, et = fit_all(resp, d)
    fx = evaluate(d, Ws, Wt)
    rs = evaluate(d, Ws, Wt, mult)
    return {"seed": seed, "b": b, "F_triple": et, "F_single": es,
            "V_triple_fixed": fx["V"][0], "V_single_fixed": fx["V"][1],
            "T_triple_fixed": fx["T"][0], "T_single_fixed": fx["T"][1],
            "V_triple_resamp": rs["V"][0], "V_single_resamp": rs["V"][1],
            "T_triple_resamp": rs["T"][0], "T_single_resamp": rs["T"][1]}


def make_draws(seed: int, B: int):
    d = _panel(seed)
    rows_F = task_rows(d["groups"]["F"])
    rows_by = {q: d["groups"][q] for q in ("V", "T")}
    G = {q: int(rows_by[q].max()) + 1 for q in rows_by}
    rng = np.random.default_rng(RNG_BASE + seed)
    jobs = []
    for b in range(B):
        pick = rng.integers(0, len(rows_F), size=len(rows_F))
        fit_rows = np.concatenate([rows_F[t] for t in pick])
        mult = {}
        for q in ("V", "T"):
            tm = np.bincount(rng.integers(0, G[q], size=G[q]), minlength=G[q]).astype(float)
            mult[q] = tm[rows_by[q]]                 # per-row multiplicity
        jobs.append((seed, b, fit_rows, mult))
    return jobs


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarise(seed: int, pt: dict, rows: list):
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("seed", "b")}
    s = {"split_seed": seed, "B": len(rows), "point": pt}
    m = {}
    for label, tri, sing, p_tri, p_sing in [
        ("F", "F_triple", "F_single", "F_triple", "F_single"),
        ("V_fixed_eval", "V_triple_fixed", "V_single_fixed", "V_triple", "V_single"),
        ("T_fixed_eval", "T_triple_fixed", "T_single_fixed", "T_triple", "T_single"),
        ("V_resampled_eval", "V_triple_resamp", "V_single_resamp", "V_triple", "V_single"),
        ("T_resampled_eval", "T_triple_resamp", "T_single_resamp", "T_triple", "T_single"),
    ]:
        margin = arr[tri] - GAMMA
        delta = arr[tri] - arr[sing]
        pm, pdlt = pt[p_tri] - GAMMA, pt[p_tri] - pt[p_sing]
        qm, qd = ci(margin), ci(delta)
        m[label] = {
            "margin_point": pm, "margin_pct95": qm,
            "margin_basic95": [2 * pm - qm[1], 2 * pm - qm[0]],
            "delta_point": pdlt, "delta_pct95": qd,
            "delta_basic95": [2 * pdlt - qd[1], 2 * pdlt - qd[0]],
            "frac_triple_gt_gamma": float(np.mean(arr[tri] > GAMMA)),
            "frac_single_gt_gamma": float(np.mean(arr[sing] > GAMMA)),
            "triple_mean_draw": float(arr[tri].mean()),
            "triple_bias_mean_minus_point": float(arr[tri].mean() - pt[p_tri]),
            "single_bias_mean_minus_point": float(arr[sing].mean() - pt[p_sing]),
        }
    s["metrics"] = m
    return s


def write_md(res: dict, path: str):
    L = []
    L.append("# E9 refit-inclusive bootstrap (reviewer items 3 and 6)\n")
    L.append(f"B = {res['B']} draws per split, 5 split seeds, gamma = {GAMMA}. "
             "Each draw resamples FIT source tasks with replacement, **refits** the three "
             "singleton maps and the three triple-removal maps on the resampled FIT, and "
             "scores them. V/T are reported two ways: *fixed-eval* (refitted maps scored on "
             "the fixed CERT/TEST: fitting variability only) and *resampled-eval* (also an "
             "independent source-task resample of CERT and of TEST in the same draw: fitting "
             "plus evaluation variability). Intervals are percentile 95%. Margin = triple "
             "error - 0.20; Delta = triple error - max singleton error on the same partition. "
             "Bias = mean over draws of the resampled-F triple error - point estimate.\n")
    L.append("## F (FIT, resampled and refitted)\n")
    L.append("| split | F margin (point) | F margin 95% | P(F triple > 0.20) | F Delta (point) | F Delta 95% | bias of F triple |")
    L.append("|---|---|---|---|---|---|---|")
    for s in res["splits"]:
        f = s["metrics"]["F"]
        L.append(f"| {s['split_seed']} | {f['margin_point']:+.4f} | [{f['margin_pct95'][0]:+.4f}, {f['margin_pct95'][1]:+.4f}] "
                 f"| {f['frac_triple_gt_gamma']:.4f} | {f['delta_point']:+.4f} | [{f['delta_pct95'][0]:+.4f}, {f['delta_pct95'][1]:+.4f}] "
                 f"| {f['triple_bias_mean_minus_point']:+.4f} |")
    L.append("\n## V and T (refitted maps)\n")
    L.append("| split | eval | V margin (point) | V margin 95% | V Delta 95% | T margin (point) | T margin 95% | T Delta 95% |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in res["splits"]:
        for ev in ("fixed_eval", "resampled_eval"):
            v, t = s["metrics"]["V_" + ev], s["metrics"]["T_" + ev]
            L.append(f"| {s['split_seed']} | {ev.replace('_', '-')} | {v['margin_point']:+.4f} "
                     f"| [{v['margin_pct95'][0]:+.4f}, {v['margin_pct95'][1]:+.4f}] "
                     f"| [{v['delta_pct95'][0]:+.4f}, {v['delta_pct95'][1]:+.4f}] "
                     f"| {t['margin_point']:+.4f} | [{t['margin_pct95'][0]:+.4f}, {t['margin_pct95'][1]:+.4f}] "
                     f"| [{t['delta_pct95'][0]:+.4f}, {t['delta_pct95'][1]:+.4f}] |")
    L.append("\n## Interpretation\n")
    L.append(res["interpretation"])
    L.append(f"\nCommand: `{res['command']}`  \nRuntime: {res['runtime_seconds']:.0f} s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def interpret(res: dict) -> str:
    F = [s["metrics"]["F"] for s in res["splits"]]
    f_inc0 = [f["margin_pct95"][0] <= 0 for f in F]
    d_pos = all(f["delta_pct95"][0] > 0 for f in F)
    vt = [s["metrics"][k] for s in res["splits"] for k in ("V_resampled_eval", "T_resampled_eval")]
    vt_inc0 = sum(x["margin_pct95"][0] <= 0 for x in vt)
    bias = [f["triple_bias_mean_minus_point"] for f in F]
    fr = [f["frac_triple_gt_gamma"] for f in F]
    txt = []
    if any(f_inc0):
        txt.append(f"The refit-inclusive 95% interval for the F triple margin includes 0 on "
                   f"{sum(f_inc0)}/{len(F)} splits (fraction of draws with F triple error > 0.20 ranges "
                   f"{min(fr):.2f}-{max(fr):.2f}), so population-level infeasibility at gamma = 0.20 is "
                   "NOT established; the defensible claim is 'infeasible on the F sample' (exact for "
                   "that sample).")
    else:
        txt.append(f"The refit-inclusive 95% interval for the F triple margin excludes 0 on all {len(F)} splits "
                   f"(fraction of draws above 0.20: {min(fr):.2f}-{max(fr):.2f}).")
    txt.append(f"The mean resampled-F triple error sits {min(bias):+.4f} to {max(bias):+.4f} from the point "
               "estimate (net effect of the max over targets x contexts and the min over w under resampling; a negative value means resample-refits look easier than the original fit).")
    txt.append(("The jump Delta (triple - max singleton) is robust: its 95% interval excludes 0 on F on every split. "
                if d_pos else "The jump Delta is not uniformly separated from 0 on F. ")
               + f"With fitting and evaluation resampling, {vt_inc0}/{len(vt)} V/T margin intervals include 0.")
    return " ".join(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--out-prefix", default="refit_bootstrap")
    a = ap.parse_args()
    t0 = time.time()
    cmd = ".venv/bin/python -u experiments/e9_refit_bootstrap.py " + " ".join(sys.argv[1:])
    points, jobs = {}, []
    for seed in a.seeds:
        points[seed] = point(seed)
        print(f"point {seed}: {points[seed]}", flush=True)
        jobs += make_draws(seed, a.boot)
    _CACHE.clear()
    with Pool(min(a.workers, 6)) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(draw, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)} draws, {time.time() - t0:.0f}s", flush=True)
    res = {
        "experiment": "E9 refit-inclusive paired source-task bootstrap",
        "gamma": GAMMA, "B": a.boot, "targets": TARGETS, "rng": f"default_rng({RNG_BASE} + seed); per draw FIT then CERT then TEST",
        "interval": "percentile 95% (basic/reverse-percentile also given)",
        "eval_modes": {"fixed_eval": "refitted maps scored on fixed CERT/TEST",
                       "resampled_eval": "refitted maps scored on independent source-task resample of CERT and TEST in same draw"},
        "command": cmd,
        "splits": [summarise(s, points[s], sorted([r for r in rows if r["seed"] == s], key=lambda r: r["b"]))
                   for s in a.seeds],
        "draws": sorted(rows, key=lambda r: (r["seed"], r["b"])),
    }
    res["runtime_seconds"] = time.time() - t0
    res["interpretation"] = interpret(res)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, a.out_prefix + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    write_md(res, os.path.join(OUT, a.out_prefix + ".md"))
    print(res["interpretation"])
    print(f"done in {res['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
