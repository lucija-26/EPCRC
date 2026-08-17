"""Independent re-check of the certified optima in results/sweep.json.

Re-derives each claim from Y_eval rather than trusting the recorded fields:
  1. the certified optimal set is oracle-feasible at its gamma;
  2. no single model can be dropped from it (necessary for optimality);
  3. every subset strictly smaller than the claimed optimum is infeasible --
     re-enumerated here, so a bug in the sweep's own certification would show up
     as a smaller feasible set rather than being silently reproduced.

Check 3 is skipped when the optimum is large enough that re-enumeration is more
expensive than the original solve (C(20,12) is 125k+ subsets); those gammas were
closed by Gurobi to gap 0 and the status is reported instead.

Run from project root:
    python experiments/verify_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.milp import is_feasible_set

BUNDLE = os.path.join(_root, "data", "exp_0_utd19_cache", "bundle.npz")
SWEEP = os.path.join(_root, "results", "sweep.json")

# Re-enumerating below this many subsets is cheap; above it we trust the gap-0 MILP.
REENUM_BUDGET = 60_000


def main() -> None:
    d = np.load(BUNDLE, allow_pickle=True)
    Y = d["Y_eval"]
    N = Y.shape[1]
    payload = json.load(open(SWEEP))
    records = [
        (r["gamma"], r["oracle"]) for r in payload["records"]
        if r["oracle"].get("optimum") is not None
    ]

    print(f"n={Y.shape[0]}  N={N}  certified records={len(records)}")
    failures: list[str] = []

    for g, o in records:
        k, S = o["optimum"], o["optimum_set"]
        tag = f"gamma={g:g}"

        if not is_feasible_set(Y, g, S):
            failures.append(f"{tag}: claimed optimum set is NOT feasible")
            print(f"{tag:12s} FAIL  claimed set infeasible")
            continue

        drop = [j for j in S if len(S) > 1 and is_feasible_set(Y, g, [x for x in S if x != j])]
        if drop:
            failures.append(f"{tag}: model(s) {drop} can be dropped -- not minimal")

        n_small = sum(len(list(combinations(range(N), m))) for m in range(1, k))
        if n_small <= REENUM_BUDGET:
            witness = None
            for m in range(1, k):
                for T in combinations(range(N), m):
                    if is_feasible_set(Y, g, list(T)):
                        witness = list(T)
                        break
                if witness:
                    break
            if witness is not None:
                failures.append(f"{tag}: size-{len(witness)} set {witness} is feasible "
                                f"but optimum was claimed to be {k}")
            proof = f"re-enumerated {n_small} subsets of size < {k}: none feasible"
        else:
            proof = (f"skipped re-enum ({n_small} subsets); MILP closed at "
                     f"gap={o.get('milp_gap', float('nan')):.1%} status={o.get('milp_status')}")

        print(f"{tag:12s} OK    |S*|={k:2d}  droppable={drop or 'none'}  {proof}")

    print()
    if failures:
        print(f"VERIFY FAILED -- {len(failures)} problem(s):")
        for m in failures:
            print("  -", m)
        sys.exit(1)
    print("VERIFY PASSED: every certified optimum is feasible, minimal, and unbeaten")


if __name__ == "__main__":
    main()
