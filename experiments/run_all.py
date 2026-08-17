"""Single entry point for the EPCRC gamma sweep: every pruner + the certified optimum.

For each gamma this records two things that live in different semantics and must
not be compared naively:

  protocol   -- the greedy pruners (backward, forward, k-swap, priority queue).
                Weights are fitted on Y_fit and scored on Y_eval, so E is
                non-monotone and no pruner has an optimality guarantee.

  oracle     -- the certified minimum, which picks routing weights directly
                against Y_eval.  That is strictly more permissive, hence
                |OPT_oracle| <= |OPT_protocol| <= |any pruner|.  It is a lower
                bound on what the pruners could possibly achieve, not a rival
                algorithm.

The oracle optimum is obtained combinatorially rather than by handing the MILP
to a solver: oracle feasibility decomposes over targets into independent tiny L1
LPs, so exhausting all subsets of size 1..max_k is cheap and parallel.  The first
size at which some subset is feasible IS the optimum, because every smaller size
was enumerated and refuted.  Only if enumeration runs out of budget do we fall
back to Gurobi, seeded with the proven bound as a cardinality cut plus a warm
start -- the LP relaxation alone stalls near 2 and never closes.

Run from project root:
    python experiments/run_all.py [gammas_csv] [max_k] [n_jobs]
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.milp import certify_lower_bound, is_feasible_set, milp_min_representative_set
from epcrc.pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)

BUNDLE = os.path.join(_root, "data", "exp_0_utd19_cache", "bundle.npz")
OUT_DIR = os.path.join(_root, "results")
OUT_FILE = os.path.join(OUT_DIR, "sweep.json")

DEFAULT_GAMMAS = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
CERT_TIME_LIMIT_S = 1800.0
MILP_TIME_LIMIT_S = 600.0

# Each entry builds a pruner from (coverage functional, gamma).  Every one of
# these returns a protocol-feasible set; they differ only in the search.
#
# Both forward variants are reported, and named for what they actually are:
#   forward       -- the literal paper-4.2 rule, stops at the first feasible
#                    prefix.  Returns non-minimal sets (up to 5 droppable models
#                    on UTD19).  This is the variant backward has always beaten.
#   forward_trim  -- forward, then the same lowest-E single-deletion trim that
#                    backward uses, so it ends minimal.  It is a hybrid, not
#                    forward selection, and only this variant beats backward.
# Collapsing the two into one row called "forward" is what made the ordering
# look like it had flipped, so keep them distinct.
PRUNERS = {
    "forward": lambda c, g: ForwardSelectionPruner(c, g, cleanup=False),
    "forward_trim": lambda c, g: ForwardSelectionPruner(c, g),
    "backward": lambda c, g: BackwardEliminationPruner(c, g),
    "backward_kswap2": lambda c, g: BackwardKSwapPruner(c, g, max_swap_k=2),
    "backward_kswap3": lambda c, g: BackwardKSwapPruner(c, g, max_swap_k=3),
    "pq_kswap": lambda c, g: PriorityQueuePruner(c, g, max_swap_k=2),
}


def oracle_incumbent(Y_eval: np.ndarray, gamma: float) -> Optional[List[int]]:
    """Backward elimination under *oracle* feasibility.

    The warm start handed to the MILP must be feasible in the MILP's own
    semantics, so this cannot reuse a protocol pruner's set.  Returns None when
    even the full ecosystem misses gamma.
    """
    S = list(range(Y_eval.shape[1]))
    if not is_feasible_set(Y_eval, gamma, S):
        return None
    for j in range(Y_eval.shape[1]):
        trial = [x for x in S if x != j]
        if trial and is_feasible_set(Y_eval, gamma, trial):
            S = trial
    return S


def run_pruners(Y_fit, Y_eval, names, gamma: float) -> Dict[str, dict]:
    out = {}
    for algo, factory in PRUNERS.items():
        cov = CoverageFunctional(Y_fit, Y_eval, names, metric="mean_abs")
        t0 = time.time()
        res = factory(cov, gamma).run()
        S = sorted(res.kept_set)

        # A pruner that is not minimal is not comparable to one that is, so the
        # droppable count is recorded rather than assumed to be zero.
        droppable = [
            j for j in S
            if len(S) > 1 and cov.compute_coverage(set(S) - {j})[0] <= gamma + 1e-9
        ]
        out[algo] = {
            "size": len(S),
            "set": S,
            "names": [names[j] for j in S],
            "coverage": float(res.coverage),
            "feasible": bool(res.coverage <= gamma + 1e-9),
            "n_droppable": len(droppable),
            "time_s": round(time.time() - t0, 2),
        }
        print(f"    {algo:<16} |S|={len(S):<3} E={res.coverage:8.3f} "
              f"({out[algo]['time_s']:.1f}s)"
              + (f"  [{len(droppable)} droppable]" if droppable else ""))
    return out


def certify_oracle(Y_eval, names, gamma: float, max_k: int, n_jobs: int) -> dict:
    inc = oracle_incumbent(Y_eval, gamma)
    if inc is None:
        return {"feasible": False}
    print(f"    oracle incumbent |S|={len(inc)}")

    # Never enumerate past the incumbent: refuting size |inc| - 1 already proves
    # the incumbent optimal.
    cap = min(max_k, len(inc) - 1) if len(inc) > 1 else 0
    cert = (
        certify_lower_bound(Y_eval, gamma, cap, time_limit=CERT_TIME_LIMIT_S, n_jobs=n_jobs)
        if cap >= 1 else None
    )

    rec = {
        "feasible": True,
        "incumbent_size": len(inc),
        "certified_upto_k": int(cert.exhausted_upto) if cert else 0,
        "n_subsets_tested": int(cert.n_subsets_tested) if cert else 0,
        "certify_time_s": round(cert.time_s, 2) if cert else 0.0,
    }

    if cert is not None and cert.best_set is not None:
        opt_set, proof = sorted(cert.best_set), "enumeration"
    elif (cert.lower_bound if cert else 1) >= len(inc):
        opt_set, proof = sorted(inc), "enumeration"
    else:
        opt_set, proof = None, None

    if opt_set is None:
        lower = cert.lower_bound if cert else 1
        print(f"    enumeration exhausted at k={cap}; MILP with min_size={lower}")
        res = milp_min_representative_set(
            Y_eval, gamma, time_limit=MILP_TIME_LIMIT_S, solver="gurobi",
            threads=n_jobs, initial_set=inc, min_size=int(lower), mip_focus=3,
        )
        rec.update(
            milp_bound=res.best_bound,
            milp_gap=res.mip_gap,
            milp_time_s=round(res.solve_time_s, 2),
            milp_status=res.status,
        )
        if res.status != 0:
            rec.update(lower_bound=int(lower), proof="unproven",
                       incumbent_set=sorted(res.kept_set))
            print(f"    UNPROVEN: bound={res.best_bound:.2f} gap={res.mip_gap:.1%}")
            return rec
        opt_set, proof = sorted(res.kept_set), "milp"

    rec.update(
        optimum=len(opt_set),
        optimum_set=opt_set,
        optimum_names=[names[j] for j in opt_set],
        lower_bound=len(opt_set),
        proof=proof,
    )
    print(f"    PROVEN OPTIMUM |S*|={len(opt_set)} -> {rec['optimum_names']}")
    return rec


def main() -> None:
    gammas = ([float(x) for x in sys.argv[1].split(",")]
              if len(sys.argv) > 1 else DEFAULT_GAMMAS)
    max_k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 2) - 2)

    d = np.load(BUNDLE, allow_pickle=True)
    Y_fit, Y_eval = d["Y_fit"], d["Y_eval"]
    names = [str(x) for x in d["model_names"]]

    print(f"UTD19  n_fit={Y_fit.shape[0]}  n_eval={Y_eval.shape[0]}  N={len(names)}")
    print(f"gammas={gammas}  max_k={max_k}  n_jobs={n_jobs}")

    t_start = time.time()
    records = []
    for g in gammas:
        print(f"\ngamma = {g:g}")
        rec = {"gamma": g, "pruners": run_pruners(Y_fit, Y_eval, names, g)}
        rec["oracle"] = certify_oracle(Y_eval, names, g, max_k, n_jobs)
        records.append(rec)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({
            "experiment": "gamma_sweep_all",
            "config": {
                "n_fit_rows": int(Y_fit.shape[0]),
                "n_eval_rows": int(Y_eval.shape[0]),
                "n_models": len(names),
                "gammas": gammas,
                "max_k": max_k,
                "metric": "mean_abs",
            },
            "model_names": names,
            "records": records,
        }, f, indent=2)
    print(f"\n[io] wrote {OUT_FILE}  ({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
