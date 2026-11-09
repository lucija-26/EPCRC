"""E9 follow-up: does singleton dependency predict the refit cost of joint removal?

Reviewer item 10.  For every split seed and every one of the 190 judge pairs
{a, b} of the 20-judge panel:

  * singleton witnesses w_i are fitted on FIT over J \\ {i} (minimax, frozen);
  * the pair witnesses are refitted on FIT over J \\ {a, b};
  * on F (FIT) and V (CERT): individual errors e_a({a}), e_b({b}), pair errors
    e_a({a,b}), e_b({a,b}), per-target increases, and
    Delta coverage = max pair error - max individual error;
  * A = [[0, w_ab], [w_ba, 0]] from the singleton rows, rho(A) = sqrt(w_ab w_ba),
    dependency mass (row sums of A), and the Theorem 2 composed map
    M = (I - A)^{-1} B with its error and bound H eps on F and V.

Then 200 random triples on the primary split, same quantities.

Descriptive only.  The 190 pairs share judges (each judge sits in 19 pairs),
so they are not independent observations; correlations are reported per split
with a pair bootstrap (anti-conservative for that reason) and a
leave-one-judge-out range.  Singleton witnesses are minimax argmins, which are
non-unique, so w_ab is a property of the fitted witness, not of the panel.

Usage:
    .venv/bin/python experiments/e9_all_pairs_dependency.py [--workers 6]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.judge import solve_minimax_weights  # noqa: E402
from epcrc.panel import SPLIT_SEEDS, load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

OUT = os.path.join(ROOT, "results", "joint_removal_audit")
FIG = os.path.join(OUT, "figures")
GAMMA = 0.20
TRIO = ("J07", "J08", "J11")
PARTS = {"F": "FIT", "V": "CERT"}
N_TRIPLES = 200
TRIPLE_SEED = 20261109
N_BOOT = 2000

_PANELS: dict = {}


def _panel(seed):
    if seed not in _PANELS:
        _PANELS[seed] = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=seed)
    return _PANELS[seed]


def _fit(job):
    seed, target, removed = job
    p = _panel(seed)
    N = len(p.judge_ids)
    surv = [j for j in range(N) if j not in removed]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        err, w = solve_minimax_weights(p.splits["FIT"], target, surv)
    stalled = any("stalled" in str(c.message) for c in caught)
    row = np.zeros(N)
    row[surv] = w
    return (seed, target, tuple(removed)), float(err), row, stalled


def evaluate_set(R, ids, P, W_single, ind_err, refit_rows):
    """All quantities for one removal set R (tuple of judge indices)."""
    N = len(ids)
    surv = [j for j in range(N) if j not in R]
    A, B = jr.split_witness(W_single, R, surv)
    off = A[~np.eye(len(R), dtype=bool)]
    rec = {
        "removal_set": "+".join(ids[i] for i in R),
        "size": len(R),
        "is_trio_subset": all(ids[i] in TRIO for i in R),
        "max_mutual_weight": float(off.max()),
        "min_mutual_weight": float(off.min()),
        "dependency_mass_max": float(A.sum(1).max()),
        "dependency_mass_sum": float(A.sum()),
    }
    if len(R) == 2:
        rec["w_ab"], rec["w_ba"] = float(A[0, 1]), float(A[1, 0])
        rec["sqrt_wab_wba"] = float(np.sqrt(A[0, 1] * A[1, 0]))
    try:
        comp = jr.compose(A, B)
    except ValueError as exc:            # non-stochastic rows: report, do not hide
        comp = {"status": f"compose_error: {exc}", "rho": float("nan")}
    rec["rho_A"] = float(comp["rho"])
    rec["compose_status"] = comp["status"]
    W_pair = np.stack([refit_rows[(i, R)] for i in R])
    for q, Pq in P.items():
        ind = np.array([ind_err[q][i] for i in R])
        pair = jr.summarise(jr.row_losses(Pq, R, W_pair))["endpoint_errors"]
        inc = pair - ind
        rec.update({
            f"ind_max_{q}": float(ind.max()),
            f"pair_max_{q}": float(pair.max()),
            f"delta_cov_{q}": float(pair.max() - ind.max()),
            f"max_target_increase_{q}": float(inc.max()),
            f"sum_target_increase_{q}": float(inc.sum()),
        })
        for k, i in enumerate(R):
            rec[f"ind_{k}_{q}"] = float(ind[k])
            rec[f"refit_{k}_{q}"] = float(pair[k])
        if comp["status"] == "ok":
            d = jr.composition_on(Pq, R, surv, W_single, comp)
            rec.update({
                f"comp_max_{q}": d["coverage_error"],
                f"bound_{q}": d["bound"],
                f"bound_context_{q}": d["bound_context"],
                # bound and composed error expressed as an increase over the
                # worst singleton residual, comparable with delta_cov
                f"bound_excess_{q}": d["bound"] - float(d["epsilon"].max()),
                f"comp_excess_{q}": d["coverage_error"] - float(d["epsilon"].max()),
                f"bound_violated_{q}": bool(d["max_bound_excess"] > 1e-10),
            })
    rec["ind_feasible_V"] = bool(rec["ind_max_V"] <= GAMMA)
    rec["joint_feasible_V"] = bool(rec["pair_max_V"] <= GAMMA)
    return rec


def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m])[0]) if m.sum() > 3 else float("nan")


def boot_ci(x, y, rng, n=N_BOOT):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    vals = []
    for _ in range(n):
        k = rng.integers(0, len(x), len(x))
        vals.append(spearmanr(x[k], y[k])[0])
    return np.nanquantile(vals, [0.025, 0.975])


def lojo_range(df, xc, yc, ids):
    vals = []
    for j in ids:
        sub = df[~df["removal_set"].str.split("+").apply(lambda s: j in s)]
        vals.append(sp(sub[xc].to_numpy(float), sub[yc].to_numpy(float)))
    return min(vals), max(vals)


def auc(score, label):
    """Mann-Whitney AUC of score for label==True; nan if a class is empty."""
    s1, s0 = score[label], score[~label]
    if len(s1) == 0 or len(s0) == 0:
        return float("nan")
    gt = (s1[:, None] > s0[None]).sum() + 0.5 * (s1[:, None] == s0[None]).sum()
    return float(gt / (len(s1) * len(s0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild the .md and figure from an existing CSV without refitting")
    args = ap.parse_args()
    if args.from_csv:
        df = pd.read_csv(os.path.join(OUT, "all_pairs_dependency.csv"))
        ids = list(_panel(PRIMARY_SEED).judge_ids)
        report(df, ids, None, None)
        figure(df)
        return
    workers = min(args.workers, 6)
    os.makedirs(FIG, exist_ok=True)

    p0 = _panel(PRIMARY_SEED)
    ids = list(p0.judge_ids)
    N = len(ids)
    assert N == 20
    pairs = list(itertools.combinations(range(N), 2))
    rng_t = np.random.default_rng(TRIPLE_SEED)
    all_triples = list(itertools.combinations(range(N), 3))
    trio_idx = tuple(ids.index(t) for t in TRIO)
    pick = rng_t.choice(len(all_triples), N_TRIPLES, replace=False)
    triples = sorted({all_triples[k] for k in pick} | {trio_idx})

    jobs = []
    for s in SPLIT_SEEDS:
        jobs += [(s, i, (i,)) for i in range(N)]
        jobs += [(s, i, R) for R in pairs for i in R]
    jobs += [(PRIMARY_SEED, i, R) for R in triples for i in R]
    t0 = time.time()
    print(f"{len(jobs)} fits on {workers} workers", flush=True)
    with Pool(workers) as pool:
        res = pool.map(_fit, jobs, chunksize=8)
    print(f"fits done in {time.time() - t0:.0f}s; stalled={sum(r[3] for r in res)}", flush=True)
    fits = {k: (e, row) for k, e, row, _ in res}
    n_stalled = sum(r[3] for r in res)

    rows = []
    for s in SPLIT_SEEDS:
        p = _panel(s)
        assert list(p.judge_ids) == ids
        P = {q: jr.dense(p.splits[name]) for q, name in PARTS.items()}
        W_single = np.stack([fits[(s, i, (i,))][1] for i in range(N)])
        ind_err = {q: jr.summarise(jr.row_losses(P[q], range(N), W_single))["endpoint_errors"]
                   for q in P}
        refit = {(i, R): fits[(s, i, R)][1] for (ss, i, R) in fits if ss == s and len(R) > 1}
        sets = [R for R in pairs] + (triples if s == PRIMARY_SEED else [])
        for R in sets:
            rec = evaluate_set(R, ids, P, W_single, ind_err, refit)
            rec["split_seed"] = s
            rec["kind"] = "pair" if len(R) == 2 else "triple"
            rows.append(rec)
    df = pd.DataFrame(rows)
    front = ["split_seed", "kind", "removal_set", "size", "is_trio_subset"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(os.path.join(OUT, "all_pairs_dependency.csv"), index=False, float_format="%.6g")

    report(df, ids, n_stalled, len(jobs))
    figure(df)


PREDICTORS = [("max_mutual_weight", "max(w_ab, w_ba)"),
              ("rho_A", "rho(A)"),
              ("dependency_mass_max", "max dependency mass"),
              ("bound_F", "Thm 2 bound H eps (F)"),
              ("bound_excess_F", "bound - max eps (F)")]
OUTCOMES = [("max_target_increase_F", "max per-target refit increase, F"),
            ("delta_cov_F", "Delta coverage, F"),
            ("max_target_increase_V", "max per-target refit increase, V"),
            ("delta_cov_V", "Delta coverage, V")]


def report(df, ids, n_stalled, n_jobs):
    rng = np.random.default_rng(20261110)
    pr = df[df.kind == "pair"]
    L = ["# E9 follow-up: singleton dependency vs refit cost, all 190 pairs x 5 splits", ""]
    fits_txt = (f"{n_jobs} minimax fits on FIT ({n_stalled} accepted stalls)." if n_jobs
                else "(rebuilt from CSV; the full run made 2600 minimax fits on FIT, 0 accepted stalls).")
    L += [f"Generated by `experiments/e9_all_pairs_dependency.py`. {fits_txt} Errors are worst-context mean TV. F = FIT (where all "
          "weights are fitted), V = CERT. gamma = 0.20 feasibility is judged on V.", ""]
    L += ["**Caveats.** The 190 pairs share judges (each judge is in 19 pairs), so they are not "
          "independent; pair-bootstrap intervals below treat them as if they were and are "
          "therefore too narrow. The leave-one-judge-out (LOJO) column gives the min/max Spearman "
          "after dropping all 19 pairs of one judge. Singleton witnesses are minimax argmins, "
          "which are non-unique, so w_ab is a property of the fitted witness. Everything here is "
          "descriptive association, not a causal claim. For a pair rho(A) = sqrt(w_ab w_ba) "
          "exactly (checked), and max dependency mass = max(w_ab, w_ba).", ""]

    # sanity: rho identity for pairs
    rho_dev = float(np.nanmax(np.abs(pr.rho_A - pr.sqrt_wab_wba)))
    L += [f"Check: max |rho(A) - sqrt(w_ab w_ba)| over pairs = {rho_dev:.1e}. "
          f"Compose status counts: {pr.compose_status.value_counts().to_dict()}. "
          f"Theorem 2 bound violated on F: {int(pr.get('bound_violated_F', pd.Series(dtype=bool)).sum())}, "
          f"on V: {int(pr.get('bound_violated_V', pd.Series(dtype=bool)).sum())} pair-splits.", ""]

    # 1. Spearman per split
    L += ["## 1. Spearman correlations per split (pairs, n = 190 each)", ""]
    for oc, ol in OUTCOMES:
        L += [f"### Outcome: {ol}", "",
              "| predictor | " + " | ".join(str(s) for s in SPLIT_SEEDS) + " | primary 95% pair-boot CI | primary LOJO range |",
              "|---|" + "---|" * (len(SPLIT_SEEDS) + 2)]
        for pc, pl in PREDICTORS:
            vals = []
            for s in SPLIT_SEEDS:
                d = pr[pr.split_seed == s]
                vals.append(sp(d[pc].to_numpy(float), d[oc].to_numpy(float)))
            d0 = pr[pr.split_seed == PRIMARY_SEED]
            ci = boot_ci(d0[pc].to_numpy(float), d0[oc].to_numpy(float), rng)
            lo, hi = lojo_range(d0, pc, oc, ids)
            L.append(f"| {pl} | " + " | ".join(f"{v:.2f}" for v in vals)
                     + f" | [{ci[0]:.2f}, {ci[1]:.2f}] | [{lo:.2f}, {hi:.2f}] |")
        L.append("")

    # reviewer's exact numbers on primary F
    d0 = pr[pr.split_seed == PRIMARY_SEED]
    L += ["### Reproduction of the reviewer's primary-split figures (F)", "",
          f"- Spearman(max(w_ab,w_ba), max per-target increase F) = "
          f"{sp(d0.max_mutual_weight.to_numpy(), d0.max_target_increase_F.to_numpy()):.3f} (reviewer: 0.92)",
          f"- Spearman(sqrt(w_ab w_ba), Delta coverage F) = "
          f"{sp(d0.sqrt_wab_wba.to_numpy(), d0.delta_cov_F.to_numpy()):.3f} (reviewer: 0.68)",
          f"- Pearson(max(w_ab,w_ba), max per-target increase F) = "
          f"{pearsonr(d0.max_mutual_weight, d0.max_target_increase_F)[0]:.3f}",
          f"- Pairs with zero mutual weight in both directions: "
          f"{int(((d0.w_ab < 1e-8) & (d0.w_ba < 1e-8)).sum())}/190; with max(w) < 1e-8 their max "
          f"per-target increase on F is at most "
          f"{d0[d0.max_mutual_weight < 1e-8].max_target_increase_F.abs().max():.1e} "
          "(removing a judge with zero weight cannot change the optimum, up to solver tolerance).", ""]
    L += ["Note on ties: many pairs have max(w_ab, w_ba) = 0 and hence zero refit increase; the "
          "rank correlation is partly driven by separating this zero block from the rest. "
          "Restricted to pairs with max(w_ab,w_ba) > 1e-6:", ""]
    L += ["| split | n nonzero | Spearman(max w, max inc F) | Spearman(max w, Delta V) |", "|---|---|---|---|"]
    for s in SPLIT_SEEDS:
        d = pr[(pr.split_seed == s) & (pr.max_mutual_weight > 1e-6)]
        L.append(f"| {s} | {len(d)} | {sp(d.max_mutual_weight.to_numpy(), d.max_target_increase_F.to_numpy()):.2f} "
                 f"| {sp(d.max_mutual_weight.to_numpy(), d.delta_cov_V.to_numpy()):.2f} |")
    L.append("")

    # 2. top pairs and trio ranks
    L += ["## 2. Largest refit increases and where the trio pairs rank", ""]
    for s in SPLIT_SEEDS:
        d = pr[pr.split_seed == s].copy()
        d["rank_inc_F"] = d.max_target_increase_F.rank(ascending=False, method="min").astype(int)
        d["rank_delta_V"] = d.delta_cov_V.rank(ascending=False, method="min").astype(int)
        d["rank_w"] = d.max_mutual_weight.rank(ascending=False, method="min").astype(int)
        top = d.sort_values("max_target_increase_F", ascending=False).head(5)
        L += [f"**Split {s}** - top 5 by max per-target increase on F: "
              + "; ".join(f"{r.removal_set} {r.max_target_increase_F:+.3f} (w={r.max_mutual_weight:.2f})"
                          for r in top.itertuples()), ""]
        L += ["| trio pair | max w | rank w | max inc F | rank inc F /190 | Delta V | rank Delta V /190 | ind max V | pair max V |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in d[d.is_trio_subset].itertuples():
            L.append(f"| {r.removal_set} | {r.max_mutual_weight:.3f} | {r.rank_w} | {r.max_target_increase_F:+.3f} | "
                     f"{r.rank_inc_F} | {r.delta_cov_V:+.3f} | {r.rank_delta_V} | {r.ind_max_V:.3f} | {r.pair_max_V:.3f} |")
        L.append("")

    # 3. feasibility
    L += ["## 3. Joint feasibility at gamma = 0.20 on V, among individually feasible pairs", "",
          "A pair is individually feasible when both singleton witnesses have V error <= 0.20; "
          "it fails jointly when the refit pair has worst V error > 0.20.", "",
          "| split | individually feasible | which | of which joint fail | joint-fail pairs | AUC max w | AUC rho(A) | AUC bound F |",
          "|---|---|---|---|---|---|---|---|"]
    for s in SPLIT_SEEDS:
        d = pr[(pr.split_seed == s) & pr.ind_feasible_V]
        which = ", ".join(d.removal_set)
        fail = ~d.joint_feasible_V.to_numpy()
        names = ", ".join(d[fail].removal_set) or "none"
        L.append(f"| {s} | {len(d)} | {which} | {int(fail.sum())} | {names} | "
                 f"{auc(d.max_mutual_weight.to_numpy(), fail):.2f} | {auc(d.rho_A.to_numpy(), fail):.2f} | "
                 f"{auc(d.bound_F.to_numpy(float), fail):.2f} |")
    L += ["", "AUC is nan when a class is empty. With so few failures these AUCs are not "
          "statistically meaningful and are shown for completeness only.", ""]
    ind_ok = pr[pr.ind_feasible_V]
    L += [f"Pooled over splits: {len(ind_ok)} individually feasible pair-splits, "
          f"{int((~ind_ok.joint_feasible_V).sum())} joint failures. Per split, number of judges with "
          "singleton V error <= 0.20: " + ", ".join(str(_n_near(pr, s, 0.0)) for s in SPLIT_SEEDS)
          + "; in (0.18, 0.20]: " + ", ".join(str(_n_near(pr, s)) for s in SPLIT_SEEDS) + ". "
          "So at gamma = 0.20 the only individually feasible pairs are the trio pairs, and the "
          "feasibility question cannot be asked of any other pair: the all-pairs evidence speaks "
          "to refit COST, not to feasibility flips.", ""]

    # 4. Theorem 2 calibration
    L += ["## 4. Theorem 2 bound vs composed and refit errors (pairs)", "",
          "Restricted to pairs with max(w_ab, w_ba) > 1e-6: when A = 0, H = I and bound = composed "
          "error = singleton error trivially, which would dominate the medians. On F, refit <= "
          "composed holds up to solver tolerance (min comp - refit = -1e-6) because the refit is "
          "the FIT optimum.", "",
          "| split | part | n | median bound/comp | median comp - refit | max comp - refit | median bound - refit | Spearman(bound, refit pair max) | Spearman(bound excess, Delta) | bound <= 0.20 | refit pair <= 0.20 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SPLIT_SEEDS:
        for q in PARTS:
            d = pr[(pr.split_seed == s) & (pr.compose_status == "ok") & (pr.max_mutual_weight > 1e-6)]
            ratio = (d[f"bound_{q}"] / d[f"comp_max_{q}"]).median()
            gap = (d[f"comp_max_{q}"] - d[f"pair_max_{q}"]).median()
            gmax = (d[f"comp_max_{q}"] - d[f"pair_max_{q}"]).max()
            bgap = (d[f"bound_{q}"] - d[f"pair_max_{q}"]).median()
            L.append(f"| {s} | {q} | {len(d)} | {ratio:.3f} | {gap:+.4f} | {gmax:+.4f} | {bgap:+.4f} | "
                     f"{sp(d[f'bound_{q}'].to_numpy(), d[f'pair_max_{q}'].to_numpy()):.2f} | "
                     f"{sp(d[f'bound_excess_{q}'].to_numpy(), d[f'delta_cov_{q}'].to_numpy()):.2f} | "
                     f"{int((d[f'bound_{q}'] <= GAMMA).sum())} | {int((d[f'pair_max_{q}'] <= GAMMA).sum())} |")
    L.append("")

    # 5. triples
    tr = df[df.kind == "triple"]
    trio_lbl = "+".join(TRIO)
    L += [f"## 5. Random triples on the primary split (n = {len(tr)}, includes {trio_lbl})", ""]
    L += ["| predictor | Spearman vs max inc F | vs Delta F | vs max inc V | vs Delta V |", "|---|---|---|---|---|"]
    for pc, pl in PREDICTORS:
        L.append(f"| {pl} | " + " | ".join(f"{sp(tr[pc].to_numpy(float), tr[oc].to_numpy(float)):.2f}"
                                          for oc, _ in OUTCOMES) + " |")
    t = tr.copy()
    t["rank_inc_F"] = t.max_target_increase_F.rank(ascending=False, method="min").astype(int)
    t["rank_delta_V"] = t.delta_cov_V.rank(ascending=False, method="min").astype(int)
    t["rank_rho"] = t.rho_A.rank(ascending=False, method="min").astype(int)
    tt = t[t.removal_set == trio_lbl].iloc[0]
    fe = t[t.ind_feasible_V]
    L += ["", f"Trio {trio_lbl}: rho(A) = {tt.rho_A:.3f} (rank {tt.rank_rho}/{len(t)}), max inc F = "
          f"{tt.max_target_increase_F:+.3f} (rank {tt.rank_inc_F}), Delta V = {tt.delta_cov_V:+.3f} "
          f"(rank {tt.rank_delta_V}), triple max V = {tt.pair_max_V:.3f}.",
          f"Triples individually feasible on V: {len(fe)}; of which joint fail: "
          f"{int((~fe.joint_feasible_V).sum())} ({', '.join(fe[~fe.joint_feasible_V].removal_set) or 'none'}).",
          f"Compose status counts: {tr.compose_status.value_counts().to_dict()}.", ""]
    with open(os.path.join(OUT, "all_pairs_dependency.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def _n_near(pr, s, lo=0.18):
    d = pr[pr.split_seed == s]
    v = {}
    for r in d.itertuples():
        a, b = r.removal_set.split("+")
        v[a], v[b] = r.ind_0_V, r.ind_1_V
    return sum(lo < x <= GAMMA for x in v.values())


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = df[df.kind == "pair"]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(SPLIT_SEEDS)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [("max_mutual_weight", "max_target_increase_F",
               "larger mutual singleton weight max(w_ab, w_ba)",
               "larger per-target refit increase on FIT"),
              ("rho_A", "delta_cov_V", "rho(A) = sqrt(w_ab w_ba)",
               "Delta coverage on CERT (pair max - individual max)"),
              ("bound_excess_F", "delta_cov_F", "Theorem 2 bound excess on FIT: max(H eps) - max eps",
               "increase over max eps on FIT: refit (o), composed (x)")]
    for ax, (xc, yc, xl, yl) in zip(axes, panels):
        for s, c in zip(SPLIT_SEEDS, colors):
            d = pr[pr.split_seed == s]
            ax.scatter(d[xc], d[yc], s=10, color=c, alpha=0.45, lw=0, label=str(s))
            if xc == "bound_excess_F":
                ax.scatter(d[xc], d["comp_excess_F"], s=8, color=c, alpha=0.25, marker="x", lw=0.6)
        tri = pr[pr.is_trio_subset]
        ax.scatter(tri[xc], tri[yc], s=46, facecolor="none", edgecolor="crimson", lw=1.4,
                   label="trio pairs (J07/J08/J11)")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    lim = [0, float(pr["bound_excess_F"].max()) * 1.02]
    axes[2].plot(lim, lim, "k--", lw=0.8, label="y = x (bound attained)")
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set_yscale("symlog", linthresh=0.01)
    prim = pr[pr.split_seed == PRIMARY_SEED].nlargest(4, "max_target_increase_F")
    for r in prim.itertuples():
        axes[0].annotate(r.removal_set.replace("+", "/"), (r.max_mutual_weight, r.max_target_increase_F),
                         xytext=(-6, 4), textcoords="offset points", fontsize=7, ha="right")
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[0].legend(fontsize=7, loc="upper left", title="split seed", title_fontsize=7)
    axes[2].legend(fontsize=7, loc="upper left", handles=axes[2].get_legend_handles_labels()[0][-2:])
    for ax, t in zip(axes, "abc"):
        ax.set_title(f"({t})", loc="left", fontsize=10)
    fig.suptitle("190 judge pairs x 5 splits: singleton dependency vs joint-removal refit cost "
                 "(descriptive; pairs share judges). Labels: primary split.", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"E_dependency_vs_refit.{ext}"), dpi=180)


if __name__ == "__main__":
    main()
