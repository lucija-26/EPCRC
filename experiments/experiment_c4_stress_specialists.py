"""Experiment C4 -- stress specialists and multi-context robust selection.

Claim C4 (plan section 5): selecting a physical panel using only clean items may
retire judges that are redundant on average but distinctive under position
swaps, verbosity perturbations, rubric changes or a hidden reference.  Selecting
against the worst context instead should retain a small number of such *stress
specialists* and reduce worst-context failure.

The experiment isolates one variable: **which contexts the selector is allowed
to see**.  Everything else is held fixed -- same panel, same greedy backward
selector, same locked TEST split, and every arm is scored on all seven
contexts, so the headline numbers are directly comparable:

    arm              selects on      fits weights on   scored on
    robust           all contexts    all contexts      all contexts
    clean_select     I0_clean        all contexts      all contexts
    clean_pipeline   I0_clean        I0_clean          all contexts

``clean_select`` exists to separate the selection mistake from the fitting
mistake: it is handicapped only at selection time and still receives robust
weights, so it is the *generous* version of the clean-only baseline.
``clean_pipeline`` is what someone who never thought about contexts would
actually deploy.

Weights are never fitted on the split they are scored on, and TEST is only ever
read.  Split seeds are varied for free (plan section 34.4) because the cached
score blocks fix what each judge saw; only the partition moves.

Usage:

    python -u experiments/experiment_c4_stress_specialists.py --panel core20
    python -u experiments/experiment_c4_stress_specialists.py --panel core20 \
        --split-seeds 20260818 --budgets 4,6,8,10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import (
    JudgeCoverageFunctional,
    JudgeResponses,
    context_errors,
    solve_minimax_weights,
    total_variation,
)
from epcrc.panel import PANELS, SCORES, SPLIT_SEEDS, Panel, load_panel, scores_dir
from epcrc.rewardbench import PRIMARY_SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "c4_stress_specialists.json")

# The unperturbed control context.  Every other registered context differs from
# it in exactly one respect, which is what makes a failure attributable.
CLEAN = "I0_clean"

# A judge counts as a stress specialist when deleting it costs at least this
# much more once the stress contexts are allowed to count.  0.05 TV is the
# smallest gap that is not split-seed noise on the observed panels; it is
# declared here rather than tuned per panel.
SPECIALIST_MARGIN = 0.05


# --------------------------------------------------------------------------
# restricting the context axis
# --------------------------------------------------------------------------

def only_contexts(responses: JudgeResponses, names: Sequence[str]) -> JudgeResponses:
    """The same responses seen through a subset of contexts, order preserved."""
    wanted = set(names)
    missing = wanted - set(responses.context_names)
    if missing:
        raise ValueError(f"unknown contexts {sorted(missing)}")
    keep = [c for c, name in enumerate(responses.context_names) if name in wanted]
    return JudgeResponses(
        [responses.blocks[c] for c in keep],
        [responses.context_names[c] for c in keep],
    )


# --------------------------------------------------------------------------
# redundancy diagnostics and the specialist definition
# --------------------------------------------------------------------------

def loo_redundancy(panel: Panel, contexts: Sequence[str]) -> Dict[str, dict]:
    """Cost of deleting each judge, when only `contexts` are allowed to count.

    Weights are fitted on FIT and the cost is read off CERT, so the number is a
    held-out redundancy rather than a training residual, and TEST stays locked.
    """
    fit = only_contexts(panel.splits["FIT"], contexts)
    cert = only_contexts(panel.splits["CERT"], contexts)

    out: Dict[str, dict] = {}
    for j, judge_id in enumerate(panel.judge_ids):
        kept = [i for i in range(panel.N) if i != j]
        _, w = solve_minimax_weights(fit, j, kept)
        errs = context_errors(cert, j, kept, w)
        out[judge_id] = {
            "loo_worst_context_tv": float(errs.max()),
            "binding_context": list(cert.context_names)[int(errs.argmax())],
            "per_context": {
                name: float(e) for name, e in zip(cert.context_names, errs)
            },
        }
    return out


def per_context_loo(panel: Panel) -> Dict[str, Dict[str, float]]:
    """Deletion cost of each judge under each context *taken on its own*.

    Each entry is fitted and scored on a single context, which is what makes it
    comparable across contexts.  The joint all-context figure cannot be used for
    the specialist test, because one weight vector has to serve every context at
    once: a judge can be expensive to delete there simply because the contexts
    disagree about how to replace it, which is a different phenomenon from being
    distinctive under a perturbation.
    """
    by_context = {
        name: loo_redundancy(panel, [name]) for name in panel.context_names
    }
    return {
        judge_id: {
            name: table[judge_id]["loo_worst_context_tv"]
            for name, table in by_context.items()
        }
        for judge_id in panel.judge_ids
    }


def find_specialists(
    per_context: Dict[str, Dict[str, float]],
    loo_joint: Dict[str, dict],
    margin: float = SPECIALIST_MARGIN,
) -> Dict[str, dict]:
    """Judges that look replaceable on clean items but are not under stress.

    The *gap* is what matters, not the level.  A judge with a large clean LOO is
    simply irreplaceable, and every selector keeps it, so it is not the judge C4
    is about.  The judge C4 is about is the one a clean-only selector happily
    deletes and a perturbation then exposes.
    """
    out: Dict[str, dict] = {}
    for judge_id, row in per_context.items():
        clean = row[CLEAN]
        stress = {name: v for name, v in row.items() if name != CLEAN}
        if not stress:
            raise ValueError("C4 needs at least one context besides the clean control")
        binding = max(stress, key=lambda name: stress[name])
        gap = stress[binding] - clean
        out[judge_id] = {
            "loo_clean_tv": float(clean),
            "loo_worst_stress_tv": float(stress[binding]),
            "loo_joint_worst_context_tv": float(
                loo_joint[judge_id]["loo_worst_context_tv"]
            ),
            "stress_gap": float(gap),
            "binding_context": binding,
            "is_specialist": bool(gap >= margin),
        }
    return out


# --------------------------------------------------------------------------
# selection, restricted to a context set
# --------------------------------------------------------------------------

def backward_chain(panel: Panel, contexts: Sequence[str]) -> Dict[int, List[int]]:
    """Greedy backward elimination that may only look at `contexts`.

    At each step the judge whose removal leaves the smallest worst-judge error
    is dropped, so the chain is nested and a budget can be read off directly.
    """
    fit = only_contexts(panel.splits["FIT"], contexts)
    cov = JudgeCoverageFunctional(fit, fit, panel.judge_ids)

    current = set(range(panel.N))
    chain = {panel.N: sorted(current)}
    while len(current) > 1:
        best, best_score = None, np.inf
        for j in sorted(current):
            score, _ = cov.compute_coverage(current - {j})
            if score < best_score:
                best, best_score = j, score
        current = current - {best}
        chain[len(current)] = sorted(current)
    return chain


# --------------------------------------------------------------------------
# scoring: always on every context of the locked split
# --------------------------------------------------------------------------

def evaluate(
    panel: Panel,
    kept: Sequence[int],
    fit_contexts: Sequence[str],
    specialists: Dict[str, dict],
    eval_split: str = "TEST",
) -> Dict[str, object]:
    """Fit weights on FIT (seen through `fit_contexts`), score on all contexts.

    The asymmetry is the point.  An arm may be denied the stress contexts while
    choosing and fitting, but it is always judged on all of them, because that
    is what deployment looks like.
    """
    kept = sorted(kept)
    fit = only_contexts(panel.splits["FIT"], fit_contexts)
    ev = panel.splits[eval_split]
    all_contexts = list(ev.context_names)

    per_judge: Dict[str, Dict[str, object]] = {}
    per_context_worst = {name: -np.inf for name in all_contexts}
    agreements: List[float] = []
    clean_idx = all_contexts.index(CLEAN)

    for j, judge_id in enumerate(panel.judge_ids):
        if j in kept:
            w = np.zeros(len(kept))
            w[kept.index(j)] = 1.0
        else:
            _, w = solve_minimax_weights(fit, j, kept)

        errs = context_errors(ev, j, kept, w)
        for name, e in zip(all_contexts, errs):
            per_context_worst[name] = max(per_context_worst[name], float(e))

        for c, block in enumerate(ev.blocks):
            recon = np.tensordot(block[:, kept, :], w, axes=([1], [0]))
            truth = block[:, j, :]
            agreements.append(
                float((truth.argmax(axis=1) == recon.argmax(axis=1)).mean())
            )

        per_judge[judge_id] = {
            "worst_context_mean_tv": float(errs.max()),
            "clean_mean_tv": float(errs[clean_idx]),
            "binding_context": all_contexts[int(errs.argmax())],
            "in_physical_panel": j in kept,
        }

    kept_ids = [panel.judge_ids[j] for j in kept]
    specialist_ids = [j for j, s in specialists.items() if s["is_specialist"]]

    return {
        "kept": kept_ids,
        "k": len(kept),
        "worst_judge_worst_context_tv": float(
            max(v["worst_context_mean_tv"] for v in per_judge.values())
        ),
        "worst_judge_clean_tv": float(
            max(v["clean_mean_tv"] for v in per_judge.values())
        ),
        "mean_judge_worst_context_tv": float(
            np.mean([v["worst_context_mean_tv"] for v in per_judge.values()])
        ),
        "per_context_worst_judge_tv": {
            name: float(v) for name, v in per_context_worst.items()
        },
        "verdict_agreement": float(np.mean(agreements)),
        "specialists_kept": sorted(set(kept_ids) & set(specialist_ids)),
        "n_specialists_kept": len(set(kept_ids) & set(specialist_ids)),
        "n_specialists_total": len(specialist_ids),
        "per_judge": per_judge,
    }


# --------------------------------------------------------------------------
# one split seed
# --------------------------------------------------------------------------

ARMS = {
    # arm -> (contexts visible to the selector, contexts visible to the fitter)
    "robust": ("all", "all"),
    "clean_select": ("clean", "all"),
    "clean_pipeline": ("clean", "clean"),
}


def run_split_seed(
    panel: Panel,
    budgets: Sequence[int],
) -> Dict[str, object]:
    all_contexts = list(panel.context_names)
    clean_only = [CLEAN]

    loo_joint = loo_redundancy(panel, all_contexts)
    loo_single = per_context_loo(panel)
    specialists = find_specialists(loo_single, loo_joint)

    named = [j for j, s in specialists.items() if s["is_specialist"]]
    print(f"  specialists ({len(named)}): "
          + ", ".join(f"{j}<-{specialists[j]['binding_context']}" for j in named),
          flush=True)

    # clean_select and clean_pipeline share a selector, so the chain is built
    # once per visible context set rather than once per arm.
    chains = {
        "all": backward_chain(panel, all_contexts),
        "clean": backward_chain(panel, clean_only),
    }

    contexts_for = {"all": all_contexts, "clean": clean_only}

    arms: Dict[str, object] = {}
    for arm, (select_on, fit_on) in ARMS.items():
        chain = chains[select_on]
        rows = {}
        for k in budgets:
            rows[str(k)] = evaluate(
                panel, chain[k], contexts_for[fit_on], specialists
            )
        arms[arm] = {
            "selects_on": contexts_for[select_on],
            "fits_on": contexts_for[fit_on],
            "chain": {str(k): [panel.judge_ids[j] for j in chain[k]]
                      for k in sorted(chain)},
            "budgets": rows,
        }
        print(f"  [{arm}] " + "  ".join(
            f"k={k}:{rows[str(k)]['worst_judge_worst_context_tv']:.4f}"
            f"({rows[str(k)]['n_specialists_kept']}sp)"
            for k in budgets
        ), flush=True)

    return {
        "split_items": panel.n_items,
        "loo_joint_all_contexts": loo_joint,
        "loo_per_context": loo_single,
        "specialists": specialists,
        "arms": arms,
    }


# --------------------------------------------------------------------------
# across split seeds
# --------------------------------------------------------------------------

def summarise(
    per_seed: Dict[str, dict],
    budgets: Sequence[int],
) -> Dict[str, object]:
    """Mean, SD, min and max across split seeds (plan section 34.4).

    ``delta`` is baseline minus robust, so a positive value means the robust
    selector achieved the lower worst-context error, which is the direction C4
    predicts.
    """
    seeds = sorted(per_seed)
    out: Dict[str, object] = {}

    for arm in ("clean_select", "clean_pipeline"):
        rows = {}
        for k in budgets:
            deltas, kept_gap = [], []
            for s in seeds:
                arms = per_seed[s]["arms"]
                base = arms[arm]["budgets"][str(k)]
                rob = arms["robust"]["budgets"][str(k)]
                deltas.append(base["worst_judge_worst_context_tv"]
                              - rob["worst_judge_worst_context_tv"])
                kept_gap.append(rob["n_specialists_kept"]
                                - base["n_specialists_kept"])
            d = np.asarray(deltas, dtype=float)
            rows[str(k)] = {
                "delta_worst_context_mean": float(d.mean()),
                "delta_worst_context_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                "delta_worst_context_min": float(d.min()),
                "delta_worst_context_max": float(d.max()),
                "n_seeds_robust_better": int((d > 0).sum()),
                "n_seeds": int(d.size),
                "specialists_retained_advantage_mean": float(np.mean(kept_gap)),
            }
        out[f"robust_vs_{arm}"] = rows

    specialist_counts = [
        sum(1 for v in per_seed[s]["specialists"].values() if v["is_specialist"])
        for s in seeds
    ]
    always = set.intersection(*[
        {j for j, v in per_seed[s]["specialists"].items() if v["is_specialist"]}
        for s in seeds
    ]) if seeds else set()

    out["specialist_stability"] = {
        "count_per_seed": dict(zip(seeds, specialist_counts)),
        "specialists_in_every_seed": sorted(always),
        "binding_contexts": {
            j: sorted({per_seed[s]["specialists"][j]["binding_context"]
                       for s in seeds})
            for j in sorted(always)
        },
    }
    return out


def run(
    seed: int,
    scores: str,
    split_seeds: Sequence[int],
    budgets: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    reference = load_panel(seed, scores, split_seed=split_seeds[0])
    if CLEAN not in reference.context_names:
        raise RuntimeError(
            f"{CLEAN} is not among the scored contexts {reference.context_names}; "
            "C4 has no control condition to compare against"
        )
    if reference.N < 3:
        raise RuntimeError("C4 needs at least three judges to have anything to retire")

    if budgets is None:
        budgets = [k for k in range(2, reference.N) if k % 2 == 0] or [2]
    budgets = sorted({int(k) for k in budgets if 1 <= int(k) <= reference.N})

    per_seed: Dict[str, dict] = {}
    for split_seed in split_seeds:
        print(f"[split_seed {split_seed}]", flush=True)
        panel = reference if split_seed == split_seeds[0] else load_panel(
            seed, scores, split_seed=split_seed
        )
        per_seed[str(split_seed)] = run_split_seed(panel, budgets)

    return {
        "experiment": "C4",
        "claim": "C4",
        "seed": seed,
        "split_seeds": list(split_seeds),
        "judges": reference.judge_ids,
        "models": reference.model_ids,
        "contexts": reference.context_names,
        "clean_context": CLEAN,
        "specialist_margin": SPECIALIST_MARGIN,
        "budgets": list(budgets),
        "arms": {a: {"selects_on": s, "fits_on": f} for a, (s, f) in ARMS.items()},
        "per_seed": per_seed,
        "summary": summarise(per_seed, budgets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--panel", choices=sorted(PANELS), default=None,
                        help="registered panel to load; overrides --scores")
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--split-seeds", default="all",
                        help="'all' for the five declared seeds, or a comma list")
    parser.add_argument("--budgets", default=None,
                        help="comma list of physical panel sizes; default is every "
                             "even size below N")
    args = parser.parse_args()

    if args.panel:
        args.scores = scores_dir(args.panel)
        if args.out == OUT:
            args.out = os.path.join(
                ROOT, "results", f"c4_stress_specialists_{args.panel}.json"
            )

    split_seeds = (
        list(SPLIT_SEEDS) if args.split_seeds == "all"
        else [int(s) for s in args.split_seeds.split(",")]
    )
    budgets = (
        None if args.budgets is None
        else [int(k) for k in args.budgets.split(",")]
    )

    payload = run(args.seed, args.scores, split_seeds, budgets)

    print(f"\njudges: {payload['judges']}")
    print(f"contexts: {payload['contexts']}")
    for arm, rows in payload["summary"].items():
        if arm == "specialist_stability":
            continue
        print(f"\n{arm}  (positive delta = robust selection wins)")
        for k, row in rows.items():
            print(f"  k={k:>3}  delta {row['delta_worst_context_mean']:+.4f}"
                  f" +/- {row['delta_worst_context_sd']:.4f}"
                  f"  [{row['delta_worst_context_min']:+.4f},"
                  f" {row['delta_worst_context_max']:+.4f}]"
                  f"  {row['n_seeds_robust_better']}/{row['n_seeds']} seeds")

    stability = payload["summary"]["specialist_stability"]
    print(f"\nspecialists in every split seed: "
          f"{stability['specialists_in_every_seed'] or 'none'}")
    for judge, contexts in stability["binding_contexts"].items():
        print(f"  {judge} binds on {contexts}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
