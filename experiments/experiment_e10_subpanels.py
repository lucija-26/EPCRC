"""Experiment E10 -- does individual-to-joint failure depend on this exact panel?

Robustness to ecosystem composition, NOT independent ecosystems: every
sub-panel reuses the same judges, items and prompts.

Design, fixed before running:
  * sub-panel sizes m in {10, 12, 15}, 200 uniform random sub-panels each,
    sampled without replacement from the 20 judges with seed 20261109;
    the full 20-judge panel is included as the reference;
  * primary split 20260818; fit on FIT, audit on CERT, exactly as E0;
  * gamma grid: the plan's section 23 grid (0.02 ... 0.20) plus 0.25 and 0.30
    from the backbone grid, because small panels have few judges removable
    at 0.20;
  * for each (sub-panel, gamma): R = judges whose leave-one-out CERT error
    within the sub-panel is <= gamma; when |R| >= 2, refit every member of R
    on the sub-panel minus R and record the joint CERT error, the joint FIT
    objective (an exact optimum on FIT, so FIT failure is certified for that
    retained set) and Delta = joint - max individual.

Reported: P(joint fails | |R| >= 2) on CERT and on FIT, per m and gamma, with
the number of eligible cases; all cases are kept.

Usage:
    python -u experiments/experiment_e10_subpanels.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc.judge import solve_minimax_weights, worst_context_error  # noqa: E402
from epcrc.panel import load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

OUT = os.path.join(ROOT, "results", "e10_subpanels_core20.json")
SIZES = [10, 12, 15]
N_PANELS = 200
SAMPLE_SEED = 20261109
SPLIT_SEED = 20260818
GAMMAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

_PANEL = None


def _init():
    global _PANEL
    _PANEL = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=SPLIT_SEED)


def audit(members):
    fit, cert = _PANEL.splits["FIT"], _PANEL.splits["CERT"]
    members = sorted(members)
    loo = {}
    for i in members:
        peers = [j for j in members if j != i]
        _, w = solve_minimax_weights(fit, i, peers)
        loo[i] = worst_context_error(cert, i, peers, w)
    joint_cache = {}
    out = []
    for g in GAMMAS:
        R = tuple(i for i in members if loo[i] <= g)
        rec = {"gamma": g, "n_removable": len(R)}
        if len(R) >= 2:
            if R not in joint_cache:
                surv = [j for j in members if j not in R]
                cert_e, fit_e = [], []
                for i in R:
                    f_err, w = solve_minimax_weights(fit, i, surv)
                    fit_e.append(f_err)
                    cert_e.append(worst_context_error(cert, i, surv, w))
                joint_cache[R] = (max(cert_e), max(fit_e))
            jc, jf = joint_cache[R]
            ind = max(loo[i] for i in R)
            rec.update(joint_cert=jc, joint_fit=jf, max_individual_cert=ind,
                       delta=jc - ind, fails_cert=jc > g, fails_fit=jf > g,
                       removable=[_PANEL.judge_ids[i] for i in R])
        out.append(rec)
    return {"members": [_PANEL.judge_ids[i] for i in members],
            "loo_cert": {_PANEL.judge_ids[i]: loo[i] for i in members}, "by_gamma": out}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SAMPLE_SEED)
    jobs = [("full", 20, tuple(range(20)))]
    for m in SIZES:
        seen = set()
        while len([1 for j in jobs if j[1] == m]) < N_PANELS:
            s = tuple(sorted(rng.choice(20, size=m, replace=False).tolist()))
            if s in seen:
                continue
            seen.add(s)
            jobs.append((f"m{m}", m, s))
    with Pool(max(1, os.cpu_count() - 1), initializer=_init) as pool:
        results = pool.map(audit, [j[2] for j in jobs], chunksize=4)
    panels = [dict(r, size=j[1], label=j[0]) for j, r in zip(jobs, results)]

    summary = []
    for m in [20] + SIZES:
        for g in GAMMAS:
            recs = [b for p in panels if p["size"] == m for b in p["by_gamma"] if b["gamma"] == g]
            elig = [b for b in recs if b["n_removable"] >= 2]
            summary.append({
                "size": m, "gamma": g, "n_panels": len(recs), "n_eligible": len(elig),
                "frac_fail_cert": (float(np.mean([b["fails_cert"] for b in elig])) if elig else None),
                "frac_fail_fit": (float(np.mean([b["fails_fit"] for b in elig])) if elig else None),
                "median_delta": (float(np.median([b["delta"] for b in elig])) if elig else None),
                "mean_n_removable_given_eligible": (float(np.mean([b["n_removable"] for b in elig])) if elig else None),
            })
    pooled = [b for p in panels if p["size"] != 20 for b in p["by_gamma"] if b["n_removable"] >= 2]
    doc = {
        "experiment": "E10 random sub-panels (robustness to ecosystem composition)",
        "sizes": SIZES, "n_panels_per_size": N_PANELS, "sample_seed": SAMPLE_SEED,
        "split_seed": SPLIT_SEED, "gammas": GAMMAS, "fit": "FIT", "audit": "CERT",
        "pooled_subpanel_cases": len(pooled),
        "pooled_frac_fail_cert": float(np.mean([b["fails_cert"] for b in pooled])) if pooled else None,
        "pooled_frac_fail_fit": float(np.mean([b["fails_fit"] for b in pooled])) if pooled else None,
        "summary": summary, "panels": panels, "elapsed_seconds": time.time() - t0,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    for s in summary:
        if s["n_eligible"]:
            print(f"m={s['size']:>2} gamma={s['gamma']:.2f} eligible {s['n_eligible']:>3}/{s['n_panels']:<3} "
                  f"fail CERT {s['frac_fail_cert']:.2f}  fail FIT {s['frac_fail_fit']:.2f}  "
                  f"median Delta {s['median_delta']:+.4f}")
    print(f"pooled sub-panel cases {doc['pooled_subpanel_cases']}: fail CERT {doc['pooled_frac_fail_cert']:.3f}, "
          f"fail FIT {doc['pooled_frac_fail_fit']:.3f}; {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
