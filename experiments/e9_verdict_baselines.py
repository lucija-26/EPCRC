"""Baselines and paired bootstrap intervals for the E9 argmax-verdict agreement check.

POST HOC, SUPPORTING ANALYSIS.  Added in response to a reviewer who noted that
the R2 verdict-agreement numbers in REVIEWER_CHECKS.md (individual vs triple
removal of J07, J08, J11) had no baseline and no uncertainty.  Nothing here was
preregistered; it does not change any protocol verdict.

Verdict agreement of a reconstruction with a target judge = fraction of
(item, context) cells where argmax(reconstruction) == argmax(target).

Arms, per split seed, target t in {J07, J08, J11}, partition V (CERT) / T (TEST):
  singleton    frozen singleton witness of t (t removed alone; FIT-fitted)
  triple       frozen triple refit of t (J07, J08, J11 removed jointly; FIT-fitted)
  uniform17    uniform mean of the 17 survivors (no fitting)
  best1_agree  single survivor judge with the highest FIT verdict agreement with t
  best1_tv     single survivor judge with the lowest FIT worst-context TV to t

Weights come from results/joint_removal_audit/weights.npz; baselines are chosen
on FIT only.  Context I6_reference_hidden is byte-identical to I0_clean, so the
primary summary averages over the 6 DISTINCT contexts; the 7-context mean is
kept for continuity with REVIEWER_CHECKS.md.

Uncertainty: paired source-task cluster bootstrap (epcrc.joint_removal.
bootstrap_means), 10,000 draws; every arm and every context uses the same task
multiplicities in a given draw, so differences are paired.  Percentile 95% CIs,
unadjusted for the number of comparisons.  The five split seeds re-partition the
same item pool, so they are not independent replicates; they are reported
separately, never pooled into one interval.

Ties: np.argmax takes the first maximal class.  The fraction of target cells
whose maximum is tied (to 1e-12) is reported.

Writes results/joint_removal_audit/verdict_baselines.json and verdict_baselines.md.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc import joint_removal as jr  # noqa: E402
from epcrc.panel import SPLIT_SEEDS, load_panel, scores_dir  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED  # noqa: E402

RES = os.path.join(ROOT, "results", "joint_removal_audit")
TARGETS = ["J07", "J08", "J11"]
PARTS = {"F": "FIT", "V": "CERT", "T": "TEST"}
DUP_CONTEXT = "I6_reference_hidden"
ARMS = ["singleton", "triple", "uniform17", "best1_agree", "best1_tv"]
CONTRASTS = [("triple", "singleton"), ("triple", "uniform17"),
             ("triple", "best1_agree"), ("triple", "best1_tv")]
N_BOOT = 10_000
BOOT_BASE = 20261109


def agreement_cells(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """[C, n, 3] x [C, n, 3] -> [C, n] 0/1 argmax agreement."""
    return (pred.argmax(-1) == y.argmax(-1)).astype(float)


def main():
    wz = np.load(os.path.join(RES, "weights.npz"))
    trip = "+".join(TARGETS)
    rows, contrasts, choices = [], [], []
    for seed in SPLIT_SEEDS:
        panel = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=seed)
        ids = list(panel.judge_ids)
        assert ids == [str(x) for x in wz["judge_ids"]], "judge order mismatch"
        ctx_names = list(panel.context_names)
        distinct = [c for c, nm in enumerate(ctx_names) if nm != DUP_CONTEXT]
        dup = ctx_names.index(DUP_CONTEXT)
        P = {q: jr.dense(panel.splits[n]) for q, n in PARTS.items()}
        for q in ("F", "V", "T"):
            assert np.array_equal(P[q][:, 0], P[q][:, dup]), "I6 no longer equals I0"
        rem = [ids.index(t) for t in TARGETS]
        surv = [j for j in range(len(ids)) if j not in rem]

        for t in TARGETS:
            i = ids.index(t)
            # ---- FIT-only baseline selection ----
            yF = P["F"][i]
            fit_agree = {j: float(agreement_cells(yF, P["F"][j])[distinct].mean()) for j in surv}
            fit_tv = {j: float(jr.tv(yF, P["F"][j]).mean(-1).max()) for j in surv}
            b_agree = max(surv, key=lambda j: (fit_agree[j], -j))
            b_tv = min(surv, key=lambda j: (fit_tv[j], j))
            W = {"singleton": wz[f"{seed}|{t}|{t}"], "triple": wz[f"{seed}|{trip}|{t}"]}
            u = np.zeros(len(ids)); u[surv] = 1.0 / len(surv)
            W["uniform17"] = u
            for name, j in (("best1_agree", b_agree), ("best1_tv", b_tv)):
                e = np.zeros(len(ids)); e[j] = 1.0
                W[name] = e
            assert abs(W["triple"][rem].sum()) < 1e-9, "triple refit uses a removed judge"
            choices.append({"split_seed": seed, "target": t,
                            "best1_agree": ids[b_agree], "best1_agree_fit_agreement": fit_agree[b_agree],
                            "best1_tv": ids[b_tv], "best1_tv_fit_worst_tv": fit_tv[b_tv],
                            "singleton_mass_on_other_removed": float(W["singleton"][[r for r in rem if r != i]].sum())})

            for q in ("V", "T"):
                y = P[q][i]
                cells = np.stack([agreement_cells(y, np.einsum("j,jcnd->cnd", W[a], P[q])) for a in ARMS])  # [A, C, n]
                srt = np.sort(y, axis=-1)
                tie_frac = float((np.abs(srt[..., 2] - srt[..., 1]) <= 1e-12)[distinct].mean())
                groups = np.asarray(panel.item_groups[PARTS[q]])
                bs = jr.bootstrap_means(cells, groups, N_BOOT, jr.stable_seed(BOOT_BASE, seed, t, q))  # [B, A, C]
                pt_ctx = cells.mean(-1)                                  # [A, C]
                summ = {"6ctx": (pt_ctx[:, distinct].mean(1), bs[:, :, distinct].mean(2)),
                        "7ctx": (pt_ctx.mean(1), bs.mean(2))}
                for k, a in enumerate(ARMS):
                    r = {"split_seed": seed, "partition": q, "target": t, "arm": a,
                         "n_items": int(cells.shape[-1]), "n_clusters": int(len(np.unique(groups))),
                         "target_tie_frac": tie_frac,
                         "worst_context_agreement_6ctx": float(pt_ctx[k, distinct].min())}
                    for v, (pt, b) in summ.items():
                        lo, hi = np.quantile(b[:, k], [0.025, 0.975])
                        r.update({f"agreement_{v}": float(pt[k]), f"ci_low_{v}": float(lo), f"ci_high_{v}": float(hi)})
                    rows.append(r)
                for a, b_ in CONTRASTS:
                    ka, kb = ARMS.index(a), ARMS.index(b_)
                    r = {"split_seed": seed, "partition": q, "target": t, "contrast": f"{a}-{b_}"}
                    for v, (pt, b) in summ.items():
                        d = b[:, ka] - b[:, kb]
                        lo, hi = np.quantile(d, [0.025, 0.975])
                        r.update({f"delta_{v}": float(pt[ka] - pt[kb]), f"ci_low_{v}": float(lo),
                                  f"ci_high_{v}": float(hi), f"frac_draws_ge_0_{v}": float((d >= 0).mean())})
                    contrasts.append(r)

    doc = {"label": "POST HOC, SUPPORTING: not preregistered; does not change any protocol verdict.",
           "definition": "fraction of (item, context) cells with argmax(reconstruction) == argmax(target); "
                         "primary = mean over 6 distinct contexts (I6 dropped, identical to I0)",
           "bootstrap": {"n_boot": N_BOOT, "cluster": "source task (panel.item_groups)", "paired": True,
                         "interval": "percentile 95%, unadjusted", "seed_base": BOOT_BASE},
           "baseline_selection": "FIT only", "split_seeds": SPLIT_SEEDS,
           "arms": rows, "contrasts": contrasts, "baseline_choices": choices}
    with open(os.path.join(RES, "verdict_baselines.json"), "w") as f:
        json.dump(doc, f, indent=1)
    write_md(rows, contrasts, choices)


def _rng(vals, fmt="{:.3f}"):
    vals = list(vals)
    return fmt.format(min(vals)) + "-" + fmt.format(max(vals)) if len(vals) > 1 else fmt.format(vals[0])


def write_md(rows, contrasts, choices):
    L = ["# Verdict-agreement baselines (POST HOC, SUPPORTING)\n",
         "Generated by `experiments/e9_verdict_baselines.py`. Not preregistered; added in response to a "
         "reviewer. It does not change any protocol verdict.\n",
         "Agreement = fraction of (item, context) cells where argmax of the reconstruction equals argmax of the "
         "target judge. Primary column: mean over the 6 distinct contexts (I6_reference_hidden is byte-identical "
         "to I0_clean and is dropped); the 7-context mean is shown for continuity with REVIEWER_CHECKS.md R2. "
         "Intervals: paired source-task cluster bootstrap, 10,000 draws, percentile 95%, unadjusted for "
         "multiplicity. Ranges are over the 5 split seeds, which re-partition one item pool and are not "
         "independent replicates. Baselines are selected on FIT only.\n",
         "## Point estimates (range over 5 splits)\n",
         "| Q | target | singleton | triple | uniform-17 | best-1 (FIT agree) | best-1 (FIT TV) | triple, 7-ctx |",
         "|---|---|---|---|---|---|---|---|"]
    for q in ("V", "T"):
        for t in TARGETS:
            cell = lambda a, v="6ctx": _rng(r[f"agreement_{v}"] for r in rows
                                            if r["partition"] == q and r["target"] == t and r["arm"] == a)
            L.append(f"| {q} | {t} | {cell('singleton')} | {cell('triple')} | {cell('uniform17')} | "
                     f"{cell('best1_agree')} | {cell('best1_tv')} | {cell('triple', '7ctx')} |")
    L += ["\n## Paired differences, 6 distinct contexts (per split: point [95% CI])\n",
          "| Q | target | split | triple - singleton | triple - uniform-17 | triple - best-1 (agree) | triple - best-1 (TV) |",
          "|---|---|---|---|---|---|---|"]
    for q in ("V", "T"):
        for t in TARGETS:
            for s in SPLIT_SEEDS:
                parts = []
                for a, b in CONTRASTS:
                    r = next(c for c in contrasts if c["partition"] == q and c["target"] == t
                             and c["split_seed"] == s and c["contrast"] == f"{a}-{b}")
                    parts.append(f"{r['delta_6ctx']:+.3f} [{r['ci_low_6ctx']:+.3f}, {r['ci_high_6ctx']:+.3f}]")
                L.append(f"| {q} | {t} | {s} | " + " | ".join(parts) + " |")
    L += ["\n## FIT-selected single-judge baselines\n",
          "| split | target | best-1 by FIT agreement | best-1 by FIT worst-context TV | singleton weight on the other two removed judges |",
          "|---|---|---|---|---|"]
    for c in choices:
        L.append(f"| {c['split_seed']} | {c['target']} | {c['best1_agree']} ({c['best1_agree_fit_agreement']:.3f}) | "
                 f"{c['best1_tv']} ({c['best1_tv_fit_worst_tv']:.3f}) | {c['singleton_mass_on_other_removed']:.3f} |")
    ties = [r["target_tie_frac"] for r in rows]
    L.append(f"\nTarget cells with a tied maximum (argmax takes the first class): {min(ties):.4f}-{max(ties):.4f}.\n")
    with open(os.path.join(RES, "verdict_baselines.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
