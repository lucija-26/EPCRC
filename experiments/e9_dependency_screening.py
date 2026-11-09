"""Dependency-aware screening inside backward elimination.

Plain backward elimination (experiment_e1_compression_frontier.select_backward)
scores every candidate j in the current set S by fully refitting every removed
target on S \\ {j}: |S| * (|R| + 1) LP fits per step.

Screened backward uses the paper's two results to avoid most of those fits and
returns exactly the same chain:

  * upper bound (Theorem 2, depth one): keep the current witnesses W_i over S
    for the removed targets i in R, fit ONE new witness w_j for the candidate on
    S \\ {j}, and substitute: W_i' = W_i restricted to S\\{j} + W_ij * w_j.
    This is M = (I - A)^{-1} B for the dependency block A (nilpotent here,
    because w_j uses no removed judge).  Its F error is an achievable error, so
    it upper-bounds the refit optimum UB_j.
  * lower bound (Lemma 3): refitting on a smaller support cannot beat the
    current support, so the refit error of S\\{j} is at least
    LB_j = max(E_R(S), e_j(S\\{j})), where E_R(S) is the current worst removed
    error and e_j(S\\{j}) is the candidate's own (already fitted) error.

  Exact search: evaluate the candidate with the smallest UB exactly, then every
  candidate in increasing LB order while LB_j <= best + TOL; candidates whose
  LB exceeds the best exact score cannot win and are never refitted.  Ties are
  broken by judge index exactly as select_backward does.

  Heuristic variant: drop the candidate with the smallest UB, refitting only it.

All fits are on FIT; scores are FIT worst-context errors, as in select_backward.
Writes results/joint_removal_audit/dependency_screening.json and .md.

    python -u experiments/e9_dependency_screening.py --workers 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.judge import solve_minimax_weights  # noqa: E402
from epcrc.panel import SPLIT_SEEDS, load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

RES = os.path.join(ROOT, "results", "joint_removal_audit")
# Fitted values carry up to ~1e-5 of tie-break slack and ~1e-7 of solver gap;
# a candidate is pruned only if its lower bound exceeds the best exact score
# by more than this.
TOL = 3e-5


class Fitter:
    def __init__(self, resp, P):
        self.resp, self.P, self.n_fits = resp, P, 0

    def fit(self, i, support):
        support = sorted(support)
        err, w = solve_minimax_weights(self.resp, i, support)
        self.n_fits += 1
        row = np.zeros(self.P.shape[0])
        row[support] = w
        return float(err), row

    def err(self, i, row):
        pred = np.einsum("j,jcnd->cnd", row, self.P)
        return float(jr.tv(self.P[i], pred).mean(-1).max())


def refit_all(f, removed, support):
    """Refit every removed target on `support`; returns (coverage, rows, errors)."""
    rows, errs = {}, {}
    for i in removed:
        errs[i], rows[i] = f.fit(i, support)
    return (max(errs.values()) if errs else 0.0), rows, errs


def plain_backward(resp, P):
    f = Fitter(resp, P)
    N = P.shape[0]
    S, R = set(range(N)), []
    chain, scores = {N: sorted(S)}, {N: 0.0}
    while len(S) > 1:
        best, best_score = None, np.inf
        for j in sorted(S):
            score, _, _ = refit_all(f, R + [j], S - {j})
            if score < best_score:
                best, best_score = j, score
        S.remove(best)
        R.append(best)
        chain[len(S)], scores[len(S)] = sorted(S), best_score
    return chain, scores, f.n_fits


def screened_backward(resp, P, heuristic=False, first='ub'):
    f = Fitter(resp, P)
    N = P.shape[0]
    S, R = set(range(N)), []
    W, E = {}, {}                       # current witnesses and errors of removed targets
    chain, scores = {N: sorted(S)}, {N: 0.0}
    exact_evals = []
    while len(S) > 1:
        cur = max(E.values()) if E else 0.0
        cand = {}
        for j in sorted(S):
            e_j, w_j = f.fit(j, S - {j})                    # one fit per candidate
            comp = {}
            for i in R:                                     # Theorem 2 substitution, no fit
                row = W[i].copy()
                a = row[j]
                row[j] = 0.0
                row = row + a * w_j
                comp[i] = f.err(i, row)
            ub = max([e_j] + list(comp.values()))
            lb = max(cur, e_j)
            cand[j] = (ub, lb, e_j, w_j)
        if heuristic:
            best = min(sorted(S), key=lambda j: (cand[j][0], j))
            score, rows, errs = refit_all(f, R + [best], S - {best})
            n_eval = 1
        else:
            # first exact evaluation: smallest Theorem 2 upper bound ('ub'),
            # or, as an ablation without Theorem 2, smallest Lemma 3 lower bound ('lb')
            key = 0 if first == "ub" else 1
            order_ub = min(sorted(S), key=lambda j: (cand[j][key], j))
            evaluated = {}
            score, rows, errs = refit_all(f, R + [order_ub], S - {order_ub})
            evaluated[order_ub] = (score, rows, errs)
            best_score = score
            for j in sorted(S, key=lambda j: (cand[j][1], j)):
                if j in evaluated:
                    continue
                if cand[j][1] > best_score + TOL:
                    break                                    # sorted by LB: nobody later can win
                s_j = refit_all(f, R + [j], S - {j})
                evaluated[j] = s_j
                best_score = min(best_score, s_j[0])
            # replicate select_backward's choice: smallest score, ties by index
            best = min(evaluated, key=lambda j: (evaluated[j][0], j))
            score, rows, errs = evaluated[best]
            n_eval = len(evaluated)
        exact_evals.append({"k": len(S) - 1, "candidates": len(S), "exact_evaluated": n_eval})
        S.remove(best)
        R.append(best)
        W, E = rows, errs
        chain[len(S)], scores[len(S)] = sorted(S), score
    return chain, scores, f.n_fits, exact_evals


def run(seed):
    panel = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=seed)
    resp = panel.splits["FIT"]
    P = jr.dense(resp)
    out = {"seed": seed}
    for name, fn in (("plain", lambda: plain_backward(resp, P)),
                     ("screened_exact", lambda: screened_backward(resp, P)),
                     ("screened_heuristic", lambda: screened_backward(resp, P, heuristic=True)),
                     ("screened_lb_only", lambda: screened_backward(resp, P, first="lb"))):
        t0 = time.time()
        res = fn()
        out[name] = {"chain": {int(k): v for k, v in res[0].items()},
                     "scores": {int(k): float(v) for k, v in res[1].items()},
                     "n_fits": res[2], "seconds": time.time() - t0}
        if len(res) > 3:
            out[name]["exact_evals"] = res[3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=SPLIT_SEEDS)
    args = ap.parse_args()
    t0 = time.time()
    with Pool(min(args.workers, len(args.seeds))) as pool:
        results = pool.map(run, args.seeds)

    # repo ground truth: the archived E1 chain (primary split) must match "plain"
    e1 = json.load(open(os.path.join(ROOT, "results", "e1_frontier_core20.json")))
    e1_chain = e1["methods"]["coverage_backward"]["chain"]
    ids = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=PRIMARY_SEED).judge_ids

    lines = ["# Dependency-aware screening in backward elimination (generated by experiments/e9_dependency_screening.py)\n",
             "Scores are worst-context errors on FIT. 'Exact' screening provably returns the plain chain (lower bound from Lemma 3, upper bound from Theorem 2's composition); 'heuristic' refits only the candidate with the smallest Theorem 2 upper bound.\n",
             "| Split | plain fits | screened-exact fits | saving | same chain as plain | ablation: LB-only fits (no Theorem 2) | LB-only same chain | heuristic fits | heuristic same chain | heuristic max score gap |",
             "|---|---:|---:|---:|---|---:|---|---:|---|---:|"]
    summary = []
    for r in results:
        p, s, h, lbo = r["plain"], r["screened_exact"], r["screened_heuristic"], r["screened_lb_only"]
        lsame = all(p["chain"][k] == lbo["chain"][k] for k in p["chain"])
        same = all(p["chain"][k] == s["chain"][k] for k in p["chain"])
        hsame = all(p["chain"][k] == h["chain"][k] for k in p["chain"])
        gap = max(h["scores"][k] - p["scores"][k] for k in p["scores"])
        summary.append({"seed": r["seed"], "plain_fits": p["n_fits"], "screened_fits": s["n_fits"],
                        "saving": 1 - s["n_fits"] / p["n_fits"], "same_chain": same,
                        "heuristic_fits": h["n_fits"], "heuristic_same_chain": hsame,
                        "heuristic_max_gap": gap, "lb_only_fits": lbo["n_fits"], "lb_only_same_chain": lsame,
                        "plain_seconds": p["seconds"], "screened_seconds": s["seconds"]})
        lines.append(f"| {r['seed']} | {p['n_fits']} | {s['n_fits']} | {100 * (1 - s['n_fits'] / p['n_fits']):.0f}% | "
                     f"{'yes' if same else 'NO'} | {lbo['n_fits']} | {'yes' if lsame else 'NO'} | {h['n_fits']} | {'yes' if hsame else 'no'} | {gap:+.4f} |")
    prim = [r for r in results if r["seed"] == PRIMARY_SEED]
    e1_match = None
    if prim:
        pc = prim[0]["plain"]["chain"]
        e1_match = all(sorted(ids[j] for j in pc[int(k)]) == sorted(v) for k, v in e1_chain.items() if int(k) in pc)
        lines.append(f"\nPlain chain on the primary split equals the archived E1 backward chain: {'yes' if e1_match else 'NO'}.")
    lines.append("\nExact evaluations per step (screened-exact, primary split): "
                 + ", ".join(f"k={e['k']}: {e['exact_evaluated']}/{e['candidates']}" for e in (prim[0]["screened_exact"]["exact_evals"] if prim else [])))
    doc = {"tolerance": TOL, "summary": summary, "e1_primary_chain_match": e1_match,
           "results": results, "elapsed_seconds": time.time() - t0,
           "command": "python -u " + " ".join(sys.argv)}
    json.dump(doc, open(os.path.join(RES, "dependency_screening.json"), "w"), indent=1)
    open(os.path.join(RES, "dependency_screening.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
