"""Correctness audit of every pruner + the DISCO solver, on real UTD19 data.

Checks that do not depend on knowing the right answer:

  A. DISCO fit optimality -- the returned w must actually solve
     min ||y - P w||_2 s.t. w >= 0, sum w = 1.  Verified by KKT: the reduced
     gradient P^T(Pw - y) must be equal on the support and no smaller off it.
  B. Feasibility -- every pruner's kept set must satisfy E(S) <= gamma.
  C. Minimality -- no single model may be droppable from the returned set.
  D. Theorem |OPT_milp(oracle)| <= |S_protocol|.  The oracle router may pick
     weights directly against Y_eval, so it dominates any protocol certificate.
     A pruner returning FEWER models than the certified oracle optimum is
     therefore a proof of a bug, not a better algorithm.
  E. Monotonicity of E -- reported, not asserted.  E is a max of Y_eval
     residuals from a Y_fit-fitted weight, so growing S can *raise* E.  That is
     the structural reason greedy can fail, and it is worth quantifying.

Run from project root:
    python experiments/audit_algorithms.py [gammas_csv]
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.geometry import DISCOSolver
from epcrc.pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
)

BUNDLE = os.path.join(_root, "data", "exp_0_utd19_cache", "bundle.npz")
CERT = os.path.join(_root, "results", "sweep.json")
GAMMAS = (
    [float(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"    PASS  {msg}")
    else:
        print(f"    FAIL  {msg}")
        failures.append(msg)


def audit_disco(Y_fit: np.ndarray, n_trials: int = 300, tol: float = 1e-6) -> None:
    """KKT check of the simplex-constrained least squares fit."""
    print("\n[A] DISCO simplex-LS optimality (KKT on real UTD19 columns)")
    rng = np.random.default_rng(0)
    N = Y_fit.shape[1]
    worst_viol = 0.0
    worst_sum = 0.0
    for _ in range(n_trials):
        i = int(rng.integers(N))
        k = int(rng.integers(2, N))
        peers = sorted(rng.choice([j for j in range(N) if j != i], size=k, replace=False))
        y, P = Y_fit[:, i], Y_fit[:, peers]
        _, w = DISCOSolver.solve_weights_and_distance(y, P)

        worst_sum = max(worst_sum, abs(w.sum() - 1.0))
        # Scale-free reduced gradient; multiplier mu = min over the support.
        g = P.T @ (P @ w - y)
        g = g / max(np.abs(g).max(), 1e-12)
        sup = w > 1e-9
        if not sup.any():
            continue
        mu = g[sup].min()
        viol = max(float(np.abs(g[sup] - mu).max()),
                   float(max(0.0, (mu - g[~sup]).max()) if (~sup).any() else 0.0))
        worst_viol = max(worst_viol, viol)

    print(f"    worst |sum(w) - 1|      = {worst_sum:.3e}")
    print(f"    worst KKT violation     = {worst_viol:.3e}  (normalized gradient)")
    check(worst_sum < 1e-6, "DISCO weights lie on the simplex")
    check(worst_viol < tol * 1e3, "DISCO fit satisfies KKT (is the true minimizer)")


def audit_monotonicity(cov: CoverageFunctional, n_trials: int = 200) -> None:
    print("\n[E] Monotonicity of E(S) under adding a model (diagnostic, not a bug)")
    rng = np.random.default_rng(1)
    N = cov.N
    worse = 0
    max_increase = 0.0
    for _ in range(n_trials):
        k = int(rng.integers(1, N - 1))
        S = set(int(x) for x in rng.choice(N, size=k, replace=False))
        j = int(rng.choice([x for x in range(N) if x not in S]))
        E0, _ = cov.compute_coverage(S)
        E1, _ = cov.compute_coverage(S | {j})
        if E1 > E0 + 1e-9:
            worse += 1
            max_increase = max(max_increase, E1 - E0)
    print(f"    adding a model RAISED E in {worse}/{n_trials} draws "
          f"(max increase {max_increase:.3f})")
    print("    -> E is non-monotone, so backward elimination has no optimality"
          " guarantee; this is expected, and is what the k-swap escape targets.")


def main() -> None:
    d = np.load(BUNDLE, allow_pickle=True)
    Y_fit, Y_eval = d["Y_fit"], d["Y_eval"]
    names = [str(x) for x in d["model_names"]]
    N = len(names)

    cert_opt = {}
    if os.path.exists(CERT):
        for r in json.load(open(CERT))["records"]:
            if r["oracle"].get("optimum") is not None:
                cert_opt[float(r["gamma"])] = int(r["oracle"]["optimum"])

    print(f"UTD19  n_fit={Y_fit.shape[0]}  n_eval={Y_eval.shape[0]}  N={N}")
    audit_disco(Y_fit)

    cov = CoverageFunctional(Y_fit, Y_eval, names, metric="mean_abs")
    audit_monotonicity(cov)

    # Names match experiments/run_all.py: plain `forward` is the untrimmed
    # paper-4.2 rule, `forward_trim` adds backward's single-deletion trim.  The
    # minimality check below is what distinguishes them, so both are audited.
    algos = {
        "backward": lambda c, g: BackwardEliminationPruner(c, g),
        "forward": lambda c, g: ForwardSelectionPruner(c, g, cleanup=False),
        "forward_trim": lambda c, g: ForwardSelectionPruner(c, g),
        "pq_kswap": lambda c, g: PriorityQueuePruner(c, g, max_swap_k=2),
        "backward_kswap3": lambda c, g: BackwardKSwapPruner(c, g, max_swap_k=3),
    }
    # Untrimmed forward is expected to be non-minimal -- that is the finding, not
    # a bug -- so it is exempted from the minimality assertion.
    MINIMALITY_EXEMPT = {"forward"}

    print("\n[B/C/D] Per-algorithm feasibility, minimality, and theorem check")
    summary = []
    for g in GAMMAS:
        print(f"\n  gamma = {g:g}   (certified oracle optimum: "
              f"{cert_opt.get(g, 'n/a')})")
        cov_g = CoverageFunctional(Y_fit, Y_eval, names, metric="mean_abs")
        for aname, factory in algos.items():
            res = factory(cov_g, g).run()
            S = sorted(res.kept_set)
            E, _ = cov_g.compute_coverage(set(S))

            check(E <= g + 1e-9, f"{aname}: E(S)={E:.3f} <= gamma  (|S|={len(S)})")

            droppable = [j for j in S
                         if len(S) > 1
                         and cov_g.compute_coverage(set(S) - {j})[0] <= g + 1e-9]
            if aname in MINIMALITY_EXEMPT:
                print(f"    n/a   {aname}: {len(droppable)} droppable "
                      f"(non-minimal by design)")
            else:
                check(not droppable,
                      f"{aname}: minimal (no single droppable model)"
                      + (f" -- droppable: {[names[j] for j in droppable]}" if droppable else ""))

            if g in cert_opt:
                check(len(S) >= cert_opt[g],
                      f"{aname}: |S|={len(S)} >= oracle optimum {cert_opt[g]}")

            summary.append({"gamma": g, "algorithm": aname, "size": len(S),
                            "coverage": E, "set": [names[j] for j in S],
                            "n_droppable": len(droppable),
                            "oracle_optimum": cert_opt.get(g)})

    out = os.path.join(_root, "results", "algorithm_audit.json")
    with open(out, "w") as f:
        json.dump({"failures": failures, "summary": summary}, f, indent=2)

    print(f"\n{'='*70}")
    if failures:
        print(f"AUDIT FAILED: {len(failures)} check(s)")
        for m in failures:
            print(f"  - {m}")
    else:
        print("AUDIT PASSED: all checks green")
    print(f"[io] wrote {out}")


if __name__ == "__main__":
    main()
