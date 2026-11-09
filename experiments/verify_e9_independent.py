"""Independent re-check of E9's key numbers and of Theorem 2, without epcrc.joint_removal.

Reads the raw score blocks and split manifests directly (plain json), takes the
saved weight rows, and recomputes:

  1. singleton and triple worst-context TV on CERT for every split, compared with
     coverage_errors.csv;
  2. for the primary split's triple, Theorem 2 in EXACT rational arithmetic
     (Python Fractions; every float is a dyadic rational, so the inputs are exact):
       - H = (I - A)^{-1}, M = H B exactly (SymPy);
       - the pointwise identity p_R - M p_S = H D on 200 random CERT items x 7
         contexts, checked to be exactly zero;
       - the context-wise bound  mean_c TV(p_i, M p_S) <= sum_r H_ir mean_c TV(D_r)
         and the stated bound (H eps)_i, exactly, on all CERT items;
  3. the two quantile levels used (0.975 two-sided; 1 - 0.05/5 = 0.99 for the
     five-split band) and eq. 10's alpha bound, exactly.

CSV values carry 10 significant digits, so CSV comparisons use 1e-9.

Writes results/joint_removal_audit/independent_check.json.
"""

from __future__ import annotations

import json
import os
import random
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "joint_removal_audit")
SCORES = os.path.join(ROOT, "results", "core20", "scores")
CONTEXTS = ["I0_clean", "I1_order_swap", "I2_correctness_rubric", "I3_bias_resistant_rubric",
            "I4_verbosity_pad_a", "I5_source_tags", "I6_reference_hidden"]
JUDGES = [f"J{i:02d}" for i in range(1, 21)]
TARGETS = ["J07", "J08", "J11"]
SEEDS = [20260818, 20260819, 20260820, 20260821, 20260822]


def load_raw():
    probs, pair_ids = {}, None
    for j in JUDGES:
        for c in CONTEXTS:
            d = json.load(open(os.path.join(SCORES, f"{j}__{c}.json")))
            if pair_ids is None:
                pair_ids = d["pair_ids"]
            assert d["pair_ids"] == pair_ids
            probs[(j, c)] = np.asarray(d["probabilities"], float)
    P = np.stack([np.stack([probs[(j, c)] for c in CONTEXTS]) for j in JUDGES])  # [20,7,n,3]
    pairs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "pairs_seed20260818.jsonl"))]
    base = {p["pair_id"]: p["base_item_id"] for p in pairs}
    return P, [base[p] for p in pair_ids]


def cert_rows(base_ids, seed):
    sp_ = json.load(open(os.path.join(ROOT, "data", f"split_seed{seed}.json")))["splits"]
    cert = set(sp_["CERT"])
    return np.array([i for i, b in enumerate(base_ids) if b in cert])


def worst_ctx(P, i, w, rows):
    pred = np.einsum("j,jcnd->cnd", w, P[:, :, rows])
    return float((0.5 * np.abs(P[i][:, rows] - pred).sum(-1)).mean(-1).max())


def main():
    P, base_ids = load_raw()
    wz = np.load(os.path.join(RES, "weights.npz"))
    cov = pd.read_csv(os.path.join(RES, "coverage_errors.csv"))
    idx = {t: JUDGES.index(t) for t in TARGETS}
    out = {"recompute": []}
    for seed in SEEDS:
        rows = cert_rows(base_ids, seed)
        trip = "+".join(TARGETS)
        e_trip = max(worst_ctx(P, idx[t], wz[f"{seed}|{trip}|{t}"], rows) for t in TARGETS)
        e_ind = max(worst_ctx(P, idx[t], wz[f"{seed}|{t}|{t}"], rows) for t in TARGETS)
        ref = cov[(cov.split_seed == seed) & (cov.partition == "V") & (cov.removal_set == trip)].iloc[0]
        out["recompute"].append({"seed": seed, "n_cert_pairs": int(len(rows)),
                                 "triple_V": e_trip, "csv_triple_V": float(ref.E_refit),
                                 "max_individual_V": e_ind, "csv_max_individual_V": float(ref.E_individual),
                                 "abs_diff": max(abs(e_trip - ref.E_refit), abs(e_ind - ref.E_individual))})

    # ---- exact Theorem 2, primary split, triple, CERT ----
    seed = 20260818
    rows = cert_rows(base_ids, seed)
    rem = [idx[t] for t in TARGETS]
    surv = [j for j in range(20) if j not in rem]
    W = np.stack([wz[f"{seed}|{t}|{t}"] for t in TARGETS])            # singleton rows [3,20]
    A = sp.Matrix(3, 3, lambda a, b: sp.Rational(float(W[a, rem[b]])))
    B = sp.Matrix(3, 17, lambda a, b: sp.Rational(float(W[a, surv[b]])))
    H = (sp.eye(3) - A).inv()
    M = H * B
    Hf = [[Fraction(int(sp.fraction(H[a, b])[0]), int(sp.fraction(H[a, b])[1])) for b in range(3)] for a in range(3)]
    Mf = [[Fraction(int(sp.fraction(M[a, b])[0]), int(sp.fraction(M[a, b])[1])) for b in range(17)] for a in range(3)]
    Wf = [[Fraction(float(W[a, j])) for j in range(20)] for a in range(3)]
    Pf = lambda j, c, n, d: Fraction(float(P[j, c, n, d]))  # noqa: E731

    # identity on random items
    random.seed(0)
    sample = random.sample(list(rows), 200)
    max_abs = Fraction(0)
    for n in sample:
        for c in range(7):
            for d in range(3):
                pR = [Pf(r, c, n, d) for r in rem]
                pS = [Pf(s, c, n, d) for s in surv]
                D = [pR[a] - sum(Wf[a][j] * Pf(j, c, n, d) for j in range(20)) for a in range(3)]
                for a in range(3):
                    lhs = pR[a] - sum(Mf[a][b] * pS[b] for b in range(17))
                    rhs = sum(Hf[a][r] * D[r] for r in range(3))
                    max_abs = max(max_abs, abs(lhs - rhs))
    out["exact_identity_items_checked"] = len(sample) * 7 * 3
    out["exact_identity_max_abs"] = float(max_abs)
    out["exact_identity_is_zero"] = max_abs == 0

    # exact context-wise bound on all CERT items
    n_items = len(rows)
    comp = [[Fraction(0)] * 7 for _ in range(3)]
    resid = [[Fraction(0)] * 7 for _ in range(3)]
    for n in rows:
        for c in range(7):
            p = [[Pf(j, c, n, d) for d in range(3)] for j in range(20)]
            for a in range(3):
                pr = p[rem[a]]
                single = [sum(Wf[a][j] * p[j][d] for j in range(20)) for d in range(3)]
                compd = [sum(Mf[a][b] * p[surv[b]][d] for b in range(17)) for d in range(3)]
                resid[a][c] += Fraction(1, 2) * sum(abs(pr[d] - single[d]) for d in range(3))
                comp[a][c] += Fraction(1, 2) * sum(abs(pr[d] - compd[d]) for d in range(3))
    comp = [[x / n_items for x in row] for row in comp]
    resid = [[x / n_items for x in row] for row in resid]
    eps = [max(resid[r]) for r in range(3)]
    rows_out = []
    for a in range(3):
        comp_row = max(comp[a])
        ctx_bound = max(sum(Hf[a][r] * resid[r][c] for r in range(3)) for c in range(7))
        bound = sum(Hf[a][r] * eps[r] for r in range(3))
        rows_out.append({"target": TARGETS[a], "comp_error": float(comp_row), "context_bound": float(ctx_bound),
                         "bound": float(bound), "comp_le_ctx_bound": comp_row <= ctx_bound,
                         "ctx_bound_le_bound": ctx_bound <= bound, "bound_ge_eps": bound >= eps[a]})
    out["exact_bound_rows_primary_triple_V"] = rows_out
    cm = pd.read_csv(os.path.join(RES, "composition_endpoint_metrics.csv"))
    ref = cm[(cm.split_seed == seed) & (cm.partition == "V") & (cm.removal_set == "+".join(TARGETS))]
    out["exact_vs_csv_max_abs"] = max(
        max(abs(r["comp_error"] - ref[ref.target == r["target"]].comp_error.iloc[0]),
            abs(r["bound"] - ref[ref.target == r["target"]].bound.iloc[0])) for r in rows_out)

    # ---- quantile levels and eq. 10, exact ----
    out["two_sided_95_upper_level"] = str(1 - sp.Rational(5, 100) / 2)
    out["band_level_5_splits"] = str(1 - sp.Rational(5, 100) / 5)
    out["bonferroni_two_sided_99_levels"] = [str(sp.Rational(5, 100) / (2 * 5)), str(1 - sp.Rational(5, 100) / (2 * 5))]
    alpha = max(sum(A[a, b] for b in range(3)) for a in range(3))
    Hinf = max(sum(H[a, b] for b in range(3)) for a in range(3))
    out["eq10_alpha"] = float(alpha)
    out["eq10_H_inf"] = float(Hinf)
    out["eq10_holds_exactly"] = bool(Hinf <= 1 / (1 - alpha))

    out["all_ok"] = (max(r["abs_diff"] for r in out["recompute"]) < 1e-9 and out["exact_identity_is_zero"]
                     and all(r["comp_le_ctx_bound"] and r["ctx_bound_le_bound"] and r["bound_ge_eps"] for r in rows_out)
                     and out["exact_vs_csv_max_abs"] < 1e-9 and out["eq10_holds_exactly"])
    json.dump(out, open(os.path.join(RES, "independent_check.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
