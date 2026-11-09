"""Audit the paper's fixed J07/J08/J11 removal case on explicit split manifests.

This supplement never reselects the removed set. Every manifest explicitly
partitions the original source groups; changing a random seed on an NPZ with
stored splits is not used as a substitute for changing that partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from epcrc.judge import JudgeResponses, solve_minimax_weights
from epcrc.panel import CORE20
from supplementary.common import Dataset, load_npz, save_json, sha256_file
from supplementary.exp00_response_cache.run import load_inputs, split_assignment
from supplementary.exp01_dependency_significance.run import (
    dependency_composition, loss_cube, paired_group_bootstrap,
)

FIXED_JUDGES = ("J07", "J08", "J11")


def load_explicit_split(input_path: Path, pairs_path: Path, manifest_path: Path):
    """Validate an aligned full core20 cache, then apply this exact group split."""
    base = load_npz(input_path)
    if set(base.judge_ids) != set(CORE20) or len(base.judge_ids) != 20:
        raise ValueError("The fixed paper case requires all original 20 endpoint IDs")
    model_ids = base.metadata.get("model_ids")
    if model_ids is None or model_ids != [CORE20[j] for j in base.judge_ids]:
        raise ValueError("Cached model IDs do not match the original core20 registry")
    rows, _ = load_inputs(pairs_path, "judge")
    with np.load(input_path, allow_pickle=False) as stored:
        probabilities = np.asarray(stored["probabilities"], dtype=float)
        pair_ids = stored["pair_ids"].astype(str)
        groups = stored["groups"].astype(str)
        original_stored_assignment = stored["split"].astype(str) if "split" in stored else None
        gold = np.asarray(stored["gold"]) if "gold" in stored else None
    if pair_ids.tolist() != [r["pair_id"] for r in rows]:
        raise ValueError("Cached pair order does not match the supplied source JSONL")
    if groups.tolist() != [r["base_item_id"] for r in rows]:
        raise ValueError("Cached source groups do not match the supplied source JSONL")
    assignment = split_assignment(manifest_path, rows, "judge")
    if set(assignment) != {"FIT", "CERT", "TEST"}:
        raise ValueError("The explicit manifest must produce three nonempty splits")
    manifest = json.loads(Path(manifest_path).read_text())
    group_hashes = {
        split: hashlib.sha256("\n".join(sorted(set(groups[assignment == split]))).encode()).hexdigest()
        for split in ("FIT", "CERT", "TEST")
    }
    for split in ("CERT", "TEST"):
        if len(set(groups[assignment == split])) < 2:
            raise ValueError("At least two groups in CERT and TEST are needed for cluster bootstrap")
    metadata = dict(base.metadata)
    metadata.update({
        "source_pairs_sha256": sha256_file(pairs_path),
        "split_manifest": str(Path(manifest_path).resolve()),
        "split_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_seed": manifest.get("seed"),
        "split_seed": manifest.get("seed"),
        "split_source": "explicit_verified_manifest",
        "split_group_sha256": group_hashes,
        "input_had_stored_split": original_stored_assignment is not None,
        "input_stored_split_matches_this_manifest": (
            bool(np.array_equal(original_stored_assignment, assignment))
            if original_stored_assignment is not None else None),
        "assignment_note": "Original NPZ split, if present, is deliberately replaced by this verified manifest.",
    })
    metadata.pop("requested_split_seed", None)
    data = Dataset(
        splits={s: JudgeResponses([block[assignment == s] for block in probabilities], base.context_names)
                for s in ("FIT", "CERT", "TEST")},
        groups={s: groups[assignment == s] for s in ("FIT", "CERT", "TEST")},
        judge_ids=base.judge_ids, context_names=base.context_names, metadata=metadata,
        gold={} if gold is None else {s: gold[assignment == s] for s in ("FIT", "CERT", "TEST")})
    return data


def run_fixed_case(data: Dataset, gamma=.20, bootstrap=2000, alpha=.05, bootstrap_seed=20260818):
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1]")
    removed = [data.judge_ids.index(j) for j in FIXED_JUDGES]
    kept = sorted(set(range(len(data.judge_ids))) - set(removed))
    fit = data.splits["FIT"]
    loo = np.zeros((len(removed), fit.n_judges))
    direct = np.zeros_like(loo)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for row, target in enumerate(removed):
            peers = [j for j in range(fit.n_judges) if j != target]
            _, loo[row, peers] = solve_minimax_weights(fit, target, peers)
            _, direct[row, kept] = solve_minimax_weights(fit, target, kept)
    full_loo = np.eye(fit.n_judges)
    full_loo[removed] = loo
    dependency = dependency_composition(full_loo, removed)
    result = {
        "metadata": data.metadata,
        "gamma": gamma,
        "fixed_removed_ids": list(FIXED_JUDGES),
        "fixed_removed_indices": removed,
        "kept_ids": [data.judge_ids[j] for j in kept],
        "context_names": data.context_names,
        "removed_set_policy": "J07/J08/J11 fixed by the published case, never selected or changed using this run's errors",
        "loo_weights_fixed_rows_all_endpoints": loo.tolist(),
        "joint_refit_weights_fixed_rows_all_endpoints": direct.tolist(),
        "solver_warnings": [str(w.message) for w in caught],
        "splits": {},
        "dependency": dependency,
    }
    if dependency["status"] == "contracting":
        composed = np.zeros_like(loo)
        # dependency_composition orders R increasingly. Original core20 has the
        # same order, but reindex explicitly for a valid reordered input tensor.
        dep_order = dependency["removed_indices"]
        positions = [dep_order.index(j) for j in removed]
        composed[:, kept] = np.asarray(dependency["composed_weights"])[positions]
        inverse = np.asarray(dependency["inverse"])[np.ix_(positions, positions)]
    else:
        composed = inverse = None
    for name, responses in data.splits.items():
        individual_loss = loss_cube(responses, removed, loo)
        joint_loss = loss_cube(responses, removed, direct)
        individual = individual_loss.mean(axis=1)
        joint = joint_loss.mean(axis=1)
        individual_worst, joint_worst = individual.max(axis=0), joint.max(axis=0)
        metrics = {
            "n_pairs": len(data.groups[name]),
            "n_groups": len(set(data.groups[name])),
            "per_judge": [{"judge_id": judge, "individual_tv": float(individual_worst[i]),
                           "joint_tv": float(joint_worst[i]),
                           "joint_minus_individual": float(joint_worst[i] - individual_worst[i]),
                           "individual_context_tv": individual[:, i].tolist(),
                           "joint_context_tv": joint[:, i].tolist()}
                          for i, judge in enumerate(FIXED_JUDGES)],
            "individual_max_tv": float(individual_worst.max()),
            "joint_max_tv": float(joint_worst.max()),
            "joint_minus_individual": float(joint_worst.max() - individual_worst.max()),
            "all_individuals_within_gamma_point_estimate": bool(individual_worst.max() <= gamma),
            "joint_exceeds_gamma_point_estimate": bool(joint_worst.max() > gamma),
        }
        if composed is not None:
            composed_tv = loss_cube(responses, removed, composed).mean(axis=1)
            bound = individual @ inverse.T
            metrics["composition"] = {
                "per_judge_order": list(FIXED_JUDGES),
                "composed_context_tv": composed_tv.tolist(),
                "theorem2_bound_per_judge": (inverse @ individual_worst).tolist(),
                "tighter_contextwise_bound": bound.tolist(),
                "max_bound_violation_numerical": float(np.maximum(composed_tv - bound, 0).max()),
                "interpretation": "Empirical sufficient bound for these frozen witnesses; not a population certificate or a necessary condition.",
            }
        if name in {"CERT", "TEST"}:
            metrics["bootstrap"] = paired_group_bootstrap(
                individual_loss, joint_loss, data.groups[name], gamma, bootstrap, alpha, bootstrap_seed)
            for judge, row in zip(FIXED_JUDGES, metrics["bootstrap"]["per_removed_judge"]):
                row["judge_id"] = judge
            metrics["bootstrap"]["analysis_role"] = (
                "Validation-split reproduction of the published fixed case; descriptive diagnostic"
                if name == "CERT" else
                "Held-out audit conditional on the fixed published case and FIT-trained weights")
        result["splits"][name] = metrics
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-index", type=int, default=0,
                        help="Index of the predeclared primary manifest; all other manifests are sensitivity checks")
    parser.add_argument("--gamma", type=float, default=.20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818,
                        help="Bootstrap RNG only; never used to create or alter splits")
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new destination to preserve the original audit")
    if not 0 <= args.primary_index < len(args.split_manifests):
        parser.error("primary-index does not name a supplied split manifest")
    if args.bootstrap < 100 or not 0 < args.alpha < 1 or not 0 <= args.gamma <= 1:
        parser.error("Use bootstrap >= 100, 0 < alpha < 1 and gamma in [0,1]")
    # Validate every split before fitting any case. Duplicate partitions cannot
    # masquerade as robustness evidence, even under different filenames/seeds.
    datasets = [load_explicit_split(args.input, args.pairs, path) for path in args.split_manifests]
    identities = [json.dumps(data.metadata["split_group_sha256"], sort_keys=True) for data in datasets]
    if len(identities) != len(set(identities)):
        parser.error("Duplicate actual group partition among supplied manifests")
    report = {
        "experiment": "published_fixed_trio_case",
        "status": "running",
        "fixed_removed_ids": list(FIXED_JUDGES),
        "input_sha256": sha256_file(args.input),
        "source_pairs_sha256": sha256_file(args.pairs),
        "gamma": args.gamma,
        "primary_index": args.primary_index,
        "bootstrap_seed": args.bootstrap_seed,
        "split_reports": [],
        "interpretation": [
            "CERT addresses the validation figure; TEST is separately reported and is not interchangeable with CERT.",
            "All declared split cases are retained, including failures or nonsignificant outcomes.",
            "Overlapping split manifests are sensitivity checks, not independent samples or independent ecosystems.",
            "Only the designated primary TEST analysis is primary; no pooled five-seed p-value is produced.",
            "Both individual upper bounds <= gamma and joint lower bound > gamma are needed for the strongest threshold-crossing claim.",
            "A positive difference interval alone does not establish that both sides lie on opposite sides of gamma.",
            "Recollected responses and existing public benchmark items are not the unavailable original response cache or prospective new test data.",
        ],
    }
    for index, data in enumerate(datasets):
        result = run_fixed_case(data, args.gamma, args.bootstrap, args.alpha, args.bootstrap_seed)
        result["analysis_role"] = "primary" if index == args.primary_index else "overlapping_split_sensitivity"
        result["manifest_index"] = index
        report["split_reports"].append(result)
        save_json(args.output, report)
        print(f"split {index}: CERT joint={result['splits']['CERT']['joint_max_tv']:.6f}; "
              f"TEST joint={result['splits']['TEST']['joint_max_tv']:.6f}; "
              f"role={result['analysis_role']}", flush=True)
    report["status"] = "complete"
    save_json(args.output, report)


if __name__ == "__main__":
    main()
