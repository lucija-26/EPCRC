"""Exact (SymPy) verification of Theorem 2 for E9.

Part 1 checks the algebra of Theorem 2 symbolically for generic 2x2 and 3x3
dependency blocks:
  * H (I - A) = I for H = (I - A)^{-1};
  * if A1 + B1 = 1 then M1 = 1 for M = H B (eq. 14);
  * if p_R = A p_R + B p_S + D then p_R - M p_S = H D (eq. 15).

Part 2 converts every saved A, B (results/joint_removal_audit/matrices.npz)
to exact rationals -- every float is a dyadic rational, so this is lossless --
and recomputes H and M exactly.  It checks M >= 0, H >= I entrywise, the row
sums of M against 1 + H delta (delta = the float rows' exact drift from 1),
the spectral radius from the exact characteristic polynomial, eq. 10
(||H||_inf <= 1/(1 - alpha)), and the numpy H, M used by the experiment.

Writes results/joint_removal_audit/theorem2_exact_check.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "joint_removal_audit")


def symbolic_part():
    out = {}
    for r in (2, 3):
        s = 2
        A = sp.Matrix(r, r, lambda i, j: 0 if i == j else sp.Symbol(f"a{i}{j}", nonnegative=True))
        B = sp.Matrix(r, s, lambda i, j: sp.Symbol(f"b{i}{j}", nonnegative=True))
        I = sp.eye(r)
        H = (I - A).inv()
        # (1) inverse
        inv_ok = sp.simplify(H * (I - A) - I) == sp.zeros(r, r)
        # (2) stochasticity: impose B1 = (I - A)1 by solving for the last B column
        one_r, one_s = sp.ones(r, 1), sp.ones(s, 1)
        sub = {B[i, s - 1]: (1 - sum(A[i, j] for j in range(r)) - sum(B[i, j] for j in range(s - 1)))
               for i in range(r)}
        M = (H * B).subs(sub)
        rows_ok = sp.simplify(M * one_s - one_r) == sp.zeros(r, 1)
        # (3) identity p_R - M p_S = H D, with p_S, D generic and p_R solving the witness equation
        pS = sp.Matrix(s, 1, lambda i, _: sp.Symbol(f"ps{i}"))
        D = sp.Matrix(r, 1, lambda i, _: sp.Symbol(f"d{i}"))
        pR = H * (B * pS + D)                       # unique solution of p_R = A p_R + B p_S + D
        witness_ok = sp.simplify(pR - (A * pR + B * pS + D)) == sp.zeros(r, 1)
        ident_ok = sp.simplify(pR - H * B * pS - H * D) == sp.zeros(r, 1)
        out[f"r={r}"] = {"inverse": bool(inv_ok), "M_rows_sum_to_one": bool(rows_ok),
                         "witness_equation": bool(witness_ok), "identity_pR_minus_MpS_eq_HD": bool(ident_ok)}
    return out


def exact(x: np.ndarray) -> sp.Matrix:
    return sp.Matrix(x.shape[0], x.shape[1], lambda i, j: sp.Rational(float(x[i, j])))


def real_part():
    mats = np.load(os.path.join(RES, "matrices.npz"))
    keys = sorted({k.rsplit("|", 1)[0] for k in mats.files})
    rows = []
    for key in keys:
        A_f, B_f = mats[f"{key}|A"], mats[f"{key}|B"]
        A, B = exact(A_f), exact(B_f)
        r = A.shape[0]
        I = sp.eye(r)
        H = (I - A).inv()
        M = H * B
        delta = A * sp.ones(r, 1) + B * sp.ones(B.shape[1], 1) - sp.ones(r, 1)
        row_sums = M * sp.ones(B.shape[1], 1)
        predicted = sp.ones(r, 1) + H * delta
        lam = sp.Symbol("lam")
        charpoly = sp.Poly((lam * I - A).det(), lam)
        roots = [complex(z) for z in sp.Poly(charpoly, lam).nroots(n=30)]
        rho = max(abs(z) for z in roots)
        alpha = max(sum(A[i, j] for j in range(r)) for i in range(r))
        Hinf = max(sum(H[i, j] for j in range(r)) for i in range(r))
        H_np, M_np = mats[f"{key}|H"], mats[f"{key}|M"]
        rows.append({
            "case": key,
            "rho_exact": rho,
            "M_min_exact": float(min(M)),
            "M_nonnegative": bool(min(M) >= 0),
            "H_ge_I": bool(all((H - I)[i, j] >= 0 for i in range(r) for j in range(r))),
            "max_abs_row_drift_delta": float(max(abs(d) for d in delta)),
            "max_abs_M_rowsum_minus_1": float(max(abs(v - 1) for v in row_sums)),
            "rowsum_equals_1_plus_H_delta_exactly": bool(sp.simplify(row_sums - predicted) == sp.zeros(r, 1)),
            "alpha": float(alpha), "H_inf": float(Hinf),
            "eq10_H_inf_le_1_over_1_minus_alpha": bool(Hinf <= 1 / (1 - alpha)),
            "numpy_H_max_abs_err": float(np.abs(H_np - np.array(H.evalf(30), dtype=float)).max()),
            "numpy_M_max_abs_err": float(np.abs(M_np - np.array(M.evalf(30), dtype=float)).max()),
        })
    return rows


def main():
    sym = symbolic_part()
    real = real_part()
    doc = {
        "symbolic": sym,
        "real_cases": real,
        "all_symbolic_ok": all(all(v.values()) for v in sym.values()),
        "all_real_ok": all(r["M_nonnegative"] and r["H_ge_I"] and r["rowsum_equals_1_plus_H_delta_exactly"]
                           and r["eq10_H_inf_le_1_over_1_minus_alpha"] and r["rho_exact"] < 1
                           and r["numpy_H_max_abs_err"] < 1e-12 and r["numpy_M_max_abs_err"] < 1e-12
                           for r in real),
        "n_real_cases": len(real),
        "max_M_rowsum_error": max(r["max_abs_M_rowsum_minus_1"] for r in real),
        "max_rho": max(r["rho_exact"] for r in real),
    }
    with open(os.path.join(RES, "theorem2_exact_check.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "real_cases"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
