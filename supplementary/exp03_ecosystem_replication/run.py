"""Repeat the joint-removal audit across explicitly documented model ecosystems."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from epcrc.judge import solve_minimax_weights, total_variation
from supplementary.common import demo_dataset, load_npz, save_json, select_subset

RELATIONSHIPS = {"original_panel", "independent_ecosystem", "same_panel_benchmark_shift", "subpanel", "synthetic"}


def validate_manifest(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.ecosystems must be a nonempty list")
    names = set()
    for entry in entries:
        for key in ("name", "input", "relationship", "assembly_protocol", "model_ids", "benchmark"):
            if key not in entry:
                raise ValueError(f"Missing provenance field {key}")
        if entry["name"] in names:
            raise ValueError("Ecosystem names must be unique")
        names.add(entry["name"])
        if entry["relationship"] not in RELATIONSHIPS:
            raise ValueError(f"Unknown relationship {entry['relationship']}")
        if not entry["assembly_protocol"].strip():
            raise ValueError("Document the assembly protocol; a new name is not a new ecosystem")
        models = entry["model_ids"]
        if len(models) < 2 or len(set(models)) != len(models):
            raise ValueError("Declare at least two unique model checkpoint IDs per ecosystem")
    # A benchmark change / re-split / subset of another declared panel is not new ecology.
    for a in entries:
        if a["relationship"] == "independent_ecosystem":
            for b in entries:
                if a is not b and set(a["model_ids"]).issubset(b["model_ids"]):
                    raise ValueError(f"{a['name']} is a model subset of {b['name']}; label as subpanel or same_panel_benchmark_shift")
    return entries


def provenance_overlap(entries):
    rows = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            x, y = set(a["model_ids"]), set(b["model_ids"])
            rows.append({"first": a["name"], "second": b["name"],
                         "model_overlap_count": len(x & y), "jaccard": len(x & y) / len(x | y),
                         "same_model_set": x == y})
    return rows


def weights_for(fit, kept):
    weights = np.zeros((fit.n_judges, len(kept)))
    for j in range(fit.n_judges):
        if j in kept:
            weights[j, kept.index(j)] = 1.0
        else:
            _, weights[j] = solve_minimax_weights(fit, j, kept)
    return weights


def losses_for(responses, kept, weights):
    return np.stack([total_variation(block, np.einsum("ikq,jk->ijq", block[:, kept, :], weights))
                     for block in responses.blocks])


def worst(losses):
    return float(losses.mean(axis=1).max())


def joint_gap_interval(individual, joint, groups, repetitions, seed):
    groups = np.asarray(groups)
    if len(groups) != joint.shape[1]:
        raise ValueError("group IDs do not align with response items")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two independent base-item groups are needed for uncertainty")
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    diffs = np.empty(repetitions)
    for b in range(repetitions):
        idx = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        diffs[b] = worst(joint[:, idx]) - worst(individual[:, idx])
    return {"gap": worst(joint) - worst(individual),
            "paired_cluster_percentile_interval_95": np.quantile(diffs, [.025, .975]).tolist(),
            "bootstrap_repetitions": repetitions, "base_item_groups": len(unique),
            "scope": "pointwise conditional interval at predeclared gamma; exploratory across secondary gamma values"}


def run_ecosystem(data, gammas, ks, method="backward", bootstrap=2000, seed=0):
    fit = data.splits["FIT"]
    n = fit.n_judges
    if len(set(ks)) != len(ks) or len(set(gammas)) != len(gammas):
        raise ValueError("Duplicate k or gamma values are not separate experiments")
    if n < 2 or any(k < 1 or k > n for k in ks):
        raise ValueError("Need N >= 2 and every k between 1 and N")
    if bootstrap < 2 or any(g < 0 or g > 1 for g in gammas):
        raise ValueError("Need bootstrap >= 2 and gamma in [0, 1]")
    start = time.perf_counter()
    loo_weights = np.zeros((n, n))
    for j in range(n):
        peers = sorted(set(range(n)) - {j})
        _, w = solve_minimax_weights(fit, j, peers)
        loo_weights[j, peers] = w
    loo_losses = {split: losses_for(responses, list(range(n)), loo_weights)
                  for split, responses in data.splits.items()}
    loo_cert = loo_losses["CERT"].mean(axis=1).max(axis=0)
    audits, per_item = [], {f"loo_{split.lower()}": loss for split, loss in loo_losses.items()}
    for gamma in gammas:
        removed = np.flatnonzero(loo_cert <= gamma).tolist()
        kept = sorted(set(range(n)) - set(removed))
        row = {"gamma": gamma, "removed_indices": removed,
               "removed_ids": [data.judge_ids[j] for j in removed],
               "kept_indices": kept, "selection_split": "CERT",
               "loo_cert_errors": dict(zip(data.judge_ids, loo_cert.tolist()))}
        if not removed:
            row["status"] = "no_individually_removable_endpoints"
        elif not kept:
            row["status"] = "all_endpoints_marked_removable_no_executable_nonempty_support"
            row["joint_tv"] = None
        else:
            weights = weights_for(fit, kept)
            errors = {}
            for split, responses in data.splits.items():
                joint = losses_for(responses, kept, weights)
                individual = loo_losses[split][:, :, removed]
                errors[split] = {"joint_tv": worst(joint), "individual_max_tv": worst(individual),
                                 "joint_exceeds_gamma": worst(joint) > gamma,
                                 "all_individuals_below_gamma": worst(individual) <= gamma,
                                 "individual_errors": individual.mean(1).max(0).tolist(),
                                 "joint_errors_removed": joint[:, :, removed].mean(1).max(0).tolist()}
                per_item[f"gamma_{gamma:g}__joint_{split.lower()}"] = joint
                if split == "TEST":
                    row["test_joint_minus_individual"] = joint_gap_interval(
                        individual, joint[:, :, removed], data.groups["TEST"], bootstrap, seed)
            row.update(status="evaluated", splits=errors)
            row["test_individual_joint_failure"] = (
                errors["TEST"]["all_individuals_below_gamma"] and errors["TEST"]["joint_exceeds_gamma"])
        audits.append(row)
    frontier = []
    for k in ks:
        selection_start = time.perf_counter()
        kept = sorted(select_subset(fit, k, method=method, seed=seed))
        weights = weights_for(fit, kept)
        row = {"k": k, "method": method, "kept_indices": kept,
               "kept_ids": [data.judge_ids[j] for j in kept],
               "selection_and_fit_seconds": time.perf_counter() - selection_start}
        for split, responses in data.splits.items():
            loss = losses_for(responses, kept, weights)
            row[f"{split.lower()}_tv"] = worst(loss)
            per_item[f"k{k}__{split.lower()}"] = loss
        frontier.append(row)
    certified_choices = []
    for gamma in gammas:
        acceptable = [row for row in frontier if row["cert_tv"] <= gamma]
        chosen = min(acceptable, key=lambda row: row["k"]) if acceptable else None
        certified_choices.append({"gamma": gamma,
                                  "smallest_tested_k_accepted_on_cert": chosen["k"] if chosen else None,
                                  "frozen_test_tv": chosen["test_tv"] if chosen else None,
                                  "is_global_minimum": False})
    return {"n_judges": n, "judge_ids": data.judge_ids, "metadata": data.metadata,
            "audits": audits, "frontier": frontier, "cert_selected_budget": certified_choices,
            "total_cpu_seconds": time.perf_counter() - start}, per_item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, nargs="+", default=[.12, .15, .20, .25, .30])
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument("--method", choices=["backward", "forward", "qr", "swap1", "swap2"], default="backward")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "results.json")
    args = parser.parse_args()
    if args.demo:
        datasets = [(dict(name="synthetic_smoke", relationship="synthetic", benchmark="synthetic",
                          model_ids=[], assembly_protocol="Generated only to test execution; not independent empirical evidence."),
                     demo_dataset(seed=args.seed))]
        datasets[0][0]["model_ids"] = list(datasets[0][1].judge_ids)
        overlaps = []
    else:
        document = json.loads(args.manifest.read_text())
        entries = validate_manifest(document["ecosystems"])
        datasets = []
        for entry in entries:
            path = Path(entry["input"])
            if not path.is_absolute():
                path = args.manifest.resolve().parent / path
            data = load_npz(path, seed=args.seed)
            if len(entry["model_ids"]) != len(data.judge_ids):
                raise ValueError(f"Model IDs and cached endpoint count disagree for {entry['name']}")
            cached_models = data.metadata.get("model_ids")
            if cached_models is not None and list(cached_models) != entry["model_ids"]:
                raise ValueError(f"Model IDs are not in cached endpoint order for {entry['name']}")
            entry = {**entry, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            datasets.append((entry, data))
        overlaps = provenance_overlap(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (entry, data) in enumerate(datasets):
        ks = args.k or entry.get('budgets') or sorted({max(1, len(data.judge_ids) // 2), max(1, len(data.judge_ids) - 2), len(data.judge_ids)})
        report, losses = run_ecosystem(data, args.gamma, ks, args.method, args.bootstrap, args.seed)
        loss_path = args.output.parent / f"{args.output.stem}__ecosystem_{index}.losses.npz"
        np.savez_compressed(loss_path, **losses,
                            **{f"{s.lower()}_groups": np.asarray(g).astype(str) for s, g in data.groups.items()})
        results.append({"provenance": entry, "per_item_losses": str(loss_path), **report})
        save_json(args.output, {"experiment": "ecosystem_replication", "results": results,
                               "declared_manifest_complete": len(results) == len(datasets),
                               "model_overlap_audit": overlaps,
                               "protocol": {"primary_gamma": .20,
                                            "primary_gamma_included": any(abs(g - .20) < 1e-12 for g in args.gamma),
                                            "other_gamma_values": "exploratory tolerance sensitivity",
                                            "removable_set": "FIT weights, CERT threshold, untouched TEST evaluation",
                                            "frontier": "FIT-selected subsets; smallest tested CERT-accepted budget, not a global minimum",
                                            "independence": "Assembly claims require human provenance review; benchmark changes, subpanels and resplits do not establish independent ecosystem evidence.",
                                            "output_interface": "Three-class probabilities only. This experiment does not establish generalization to token distributions or embeddings."}})
    print(f"Saved {len(results)} ecosystem report(s) to {args.output}")


if __name__ == "__main__":
    main()
