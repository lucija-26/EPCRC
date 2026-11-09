"""Equal-budget subset search, frozen reconstruction, and paired uncertainty."""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from math import comb
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from epcrc.judge import JudgeCoverageFunctional, solve_minimax_weights, total_variation
from supplementary.common import add_data_arguments, load_dataset, save_json, select_subset


def fitted_losses(fit, evaluate, kept):
    """Return C x items x endpoints losses, including zero for retained endpoints."""
    kept = sorted(kept)
    if not kept:
        raise ValueError("At least one physical endpoint must remain")
    if len({len(b) for b in evaluate.blocks}) != 1:
        raise ValueError("Paired context analysis requires aligned items in each context")
    weights = np.zeros((fit.n_judges, len(kept)))
    for j in range(fit.n_judges):
        if j in kept:
            weights[j, kept.index(j)] = 1
        else:
            _, weights[j] = solve_minimax_weights(fit, j, kept)
    losses, agreement = [], []
    for block in evaluate.blocks:
        prediction = np.einsum("ikq,jk->ijq", block[:, kept, :], weights)
        losses.append(total_variation(block, prediction))
        agreement.append(block.argmax(-1) == prediction.argmax(-1))
    return np.stack(losses), weights, np.stack(agreement)


def metric(losses):
    return float(np.asarray(losses).mean(axis=1).max())


def exact_search(fit, k, max_evaluations=10000, max_judges=10):
    """Exhaustive FIT search; an incomplete enumeration is never an optimum."""
    n = fit.n_judges
    total = comb(n, k)
    if n > max_judges:
        return {"status": "skipped_panel_too_large", "complete": False,
                "n_subsets": total, "evaluated": 0, "optimum": None, "kept": None}
    cov = JudgeCoverageFunctional(fit, fit)
    best, best_kept, count = np.inf, None, 0
    start = time.perf_counter()
    for kept in combinations(range(n), k):
        if count >= max_evaluations:
            break
        value = cov.compute_coverage(set(kept))[0]
        count += 1
        if value < best:
            best, best_kept = float(value), list(kept)
    complete = count == total
    return {"status": "complete" if complete else "budget_exhausted", "complete": complete,
            "n_subsets": total, "evaluated": count, "kept": best_kept,
            "optimum": best if complete else None,
            "best_observed_fit_objective": best if count else None,
            "seconds": time.perf_counter() - start}


def paired_bootstrap(losses, reference, groups, n_bootstrap=2000, seed=0,
                     family_size=1, alpha=0.05):
    """Conditional paired cluster bootstrap of max(mean TV), keeping contexts together."""
    groups = np.asarray(groups)
    if len(groups) != losses.shape[1] or losses.shape != reference.shape:
        raise ValueError("Losses and base-item group IDs must align")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two independent base-item groups are needed")
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        indices = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        differences[b] = metric(losses[:, indices]) - metric(reference[:, indices])
    tail = alpha / max(1, family_size) / 2
    return {"difference_method_minus_backward": metric(losses) - metric(reference),
            "pointwise_percentile_interval": np.quantile(differences, [alpha / 2, 1 - alpha / 2]).tolist(),
            "bonferroni_percentile_interval": np.quantile(differences, [tail, 1 - tail]).tolist(),
            "family_size": family_size, "bootstrap_repetitions": n_bootstrap,
            "base_item_groups": len(unique),
            "interpretation": "Negative favors this method. Conditional on FIT/CERT and frozen subsets; not a population-over-ecosystems interval.",
            "tail_resolution_warning": n_bootstrap * tail < 10}


def run(data, ks, methods, random_repeats=5, bootstrap=2000, seed=0,
        exact_budget=10000, exact_max_judges=10):
    fit = data.splits["FIT"]
    n = fit.n_judges
    if len(set(ks)) != len(ks) or len(set(methods)) != len(methods):
        raise ValueError("Duplicate k values or method names are not separate comparisons")
    if any(k < 1 or k > n for k in ks):
        raise ValueError(f"k must lie in [1, {n}]")
    if bootstrap < 2 or random_repeats < 1 or exact_budget < 0:
        raise ValueError("Use bootstrap >= 2, random repeats >= 1, and a nonnegative exact budget")
    rows, all_losses, exact_rows = [], {}, []
    labels = []
    for method in methods:
        if method == "random":
            labels.extend((f"random_{r}", "random", seed + r) for r in range(random_repeats))
        else:
            labels.append((method, method, seed))
    if "backward" not in {m for _, m, _ in labels}:
        raise ValueError("Include backward as the predeclared paired reference")
    family_size = len(ks) * (len(labels) - 1)
    for k in ks:
        exact = exact_search(fit, k, exact_budget, exact_max_judges)
        exact_rows.append({"k": k, **exact})
        by_method = {}
        for label, method, method_seed in labels:
            start = time.perf_counter()
            kept = sorted(select_subset(fit, k, method=method, seed=method_seed))
            selection_seconds = time.perf_counter() - start
            start = time.perf_counter()
            fit_loss, weights, _ = fitted_losses(fit, fit, kept)
            # Reuse exactly the FIT weights on CERT/TEST; no eval-dependent fitting.
            evaluated, agreement = {}, {}
            for split in ("CERT", "TEST"):
                split_losses, split_agree = [], []
                for block in data.splits[split].blocks:
                    pred = np.einsum("ikq,jk->ijq", block[:, kept, :], weights)
                    split_losses.append(total_variation(block, pred))
                    split_agree.append(block.argmax(-1) == pred.argmax(-1))
                evaluated[split] = np.stack(split_losses)
                agreement[split] = np.stack(split_agree)
            loss = evaluated["TEST"]
            key = f"k{k}__{label}"
            all_losses[key] = loss
            by_method[label] = loss
            removed = sorted(set(range(n)) - set(kept))
            row = {"k": k, "method": label, "seed": method_seed,
                   "kept_indices": kept, "kept_ids": [data.judge_ids[j] for j in kept],
                   "fit_tv": metric(fit_loss), "cert_tv": metric(evaluated["CERT"]),
                   "test_tv": metric(loss), "selection_seconds": selection_seconds,
                   "fit_and_evaluation_seconds": time.perf_counter() - start,
                   "removed_verdict_agreement": float(agreement["TEST"][:, :, removed].mean()) if removed else 1.0,
                   "fit_gap_to_exact": metric(fit_loss) - exact["optimum"] if exact["complete"] else None,
                   "exact_search_complete": exact["complete"]}
            rows.append(row)
        for row in [r for r in rows if r["k"] == k and r["method"] != "backward"]:
            row["paired_test_bootstrap"] = paired_bootstrap(
                by_method[row["method"]], by_method["backward"], data.groups["TEST"],
                bootstrap, seed, family_size)
    return {"experiment": "selection_benchmarks", "metadata": data.metadata,
            "protocol": {"selection": "FIT only", "reconstruction": "global simplex minimax FIT weights",
                         "evaluation": "frozen CERT and TEST", "reference": "backward",
                         "intervals": "paired base-item cluster percentile bootstrap, with Bonferroni across k/method comparisons",
                         "exact_scope": "finite FIT objective only, not held-out or population optimality",
                         "swaps": "Fixed-cardinality exchanges; at most four accepted improvement rounds, not a guarantee of local optimality.",
                         "timing_scope": "Per requested k via a shared selector API. QR/k-medoids implementations construct their full k-chain in each call; timings are not optimized amortized frontier runtimes.",
                         "caution": "Five overlapping split seeds are sensitivity analyses, not five independent ecosystems."},
            "rows": rows, "exact_search": exact_rows}, all_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=["backward", "forward", "qr", "kmedoids", "random", "swap1", "swap2"],
                        choices=["backward", "forward", "qr", "kmedoids", "random", "swap1", "swap2"])
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--exact-budget", type=int, default=10000)
    parser.add_argument("--exact-max-judges", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "results.json")
    args = parser.parse_args()
    data = load_dataset(args)
    ks = args.k or sorted({max(1, len(data.judge_ids) // 2), max(1, len(data.judge_ids) - 2)})
    report, losses = run(data, ks, args.methods, args.random_repeats, args.bootstrap,
                         args.seed, args.exact_budget, args.exact_max_judges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_path = args.output.with_suffix(".losses.npz")
    np.savez_compressed(loss_path, **losses, test_groups=np.asarray(data.groups["TEST"]).astype(str))
    report["per_item_losses"] = str(loss_path)
    save_json(args.output, report)
    columns = ["k", "method", "fit_tv", "cert_tv", "test_tv", "selection_seconds", "fit_and_evaluation_seconds", "removed_verdict_agreement", "fit_gap_to_exact", "exact_search_complete"]
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    print(f"Saved {args.output}; real/synthetic provenance: {data.metadata}")


if __name__ == "__main__":
    main()
