"""Audit individual/joint removal on TEST, with grouped paired uncertainty.

All replacements are fitted on FIT. CERT fixes the removal set, so TEST can
evaluate a frozen decision. The bootstrap resamples base items jointly across
contexts and methods; it never treats the five overlapping splits as trials.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcrc.judge import JudgeResponses, solve_minimax_weights, total_variation


def loo_weights(fit: JudgeResponses) -> np.ndarray:
    """Row-stochastic FIT certificates, with an identically zero diagonal."""
    n = fit.n_judges
    if n < 2:
        raise ValueError("At least two judges are required")
    weights = np.zeros((n, n))
    for target in range(n):
        kept = [j for j in range(n) if j != target]
        _, weights[target, kept] = solve_minimax_weights(fit, target, kept)
    return weights


def loss_cube(responses: JudgeResponses, targets, weights: np.ndarray) -> np.ndarray:
    """Return (context, item, target) TV for full-panel replacement rows."""
    targets = list(targets)
    lengths = {len(block) for block in responses.blocks}
    if len(lengths) != 1:
        raise ValueError("Paired context bootstrap requires aligned item rows")
    if weights.shape != (len(targets), responses.n_judges):
        raise ValueError("Replacement matrix has the wrong shape")
    return np.stack([
        total_variation(block[:, targets, :], np.einsum("nmk,jm->njk", block, weights))
        for block in responses.blocks
    ])


def dependency_composition(weights: np.ndarray, removed, spectral_tol=1e-10) -> dict:
    """Compute the absorbing replacement map, never invert a closed cycle.

    For P_R = A P_R + B P_S + residual, rho(A) < 1 implies
    P_R - (I-A)^-1 B P_S = (I-A)^-1 residual. Triangle inequality
    gives a contextwise TV bound using the nonnegative inverse.
    """
    weights = np.asarray(weights, dtype=float)
    n = weights.shape[0]
    if weights.shape != (n, n) or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite square matrix")
    if np.min(weights) < -1e-9 or not np.allclose(weights.sum(axis=1), 1, atol=1e-7):
        raise ValueError("weights must be row stochastic")
    removed = sorted(set(int(i) for i in removed))
    if not removed or min(removed) < 0 or max(removed) >= n:
        raise ValueError("removed must be a nonempty valid subset")
    kept = sorted(set(range(n)) - set(removed))
    A = weights[np.ix_(removed, removed)]
    B = weights[np.ix_(removed, kept)]
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    result = {
        "removed_indices": removed,
        "kept_indices": kept,
        "A_removed_to_removed": A.tolist(),
        "B_removed_to_retained": B.tolist(),
        "spectral_radius": rho,
        "max_internal_row_mass": float(A.sum(axis=1).max()),
        "condition_number_I_minus_A": None,
        "status": "not_contracting",
        "inverse": None,
        "composed_weights": None,
    }
    if not kept or rho >= 1 - spectral_tol:
        result["reason"] = "rho(A) is numerically at least one, or no endpoint remains"
        return result
    system = np.eye(len(removed)) - A
    condition = float(np.linalg.cond(system))
    result["condition_number_I_minus_A"] = condition
    inverse = np.linalg.solve(system, np.eye(len(removed)))
    composed = inverse @ B
    if (inverse.min() < -1e-7 or composed.min() < -1e-7
            or not np.allclose(composed.sum(axis=1), 1, atol=1e-6)):
        result["status"] = "numerically_unstable"
        result["reason"] = "The computed absorbing map failed probability checks"
        return result
    result.update({
        "status": "contracting",
        "inverse": inverse.tolist(),
        "inverse_row_sums": inverse.sum(axis=1).tolist(),
        "composed_weights": composed.tolist(),
        "composed_min_weight": float(composed.min()),
        "composed_max_row_sum_error": float(np.abs(composed.sum(axis=1) - 1).max()),
    })
    return result


def _interval(samples: np.ndarray, alpha: float) -> list:
    return np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist()


def paired_group_bootstrap(individual: np.ndarray, joint: np.ndarray, groups,
                           gamma: float, repetitions=2000, alpha=.05, seed=0) -> dict:
    """Conditional inference for a frozen FIT/CERT decision.

The primary interval is a simultaneous sup-norm bootstrap band over both
methods and all primitive context/target means. Maxima and differences of
maxima are bounded using this band. Paired percentile intervals are also
reported, explicitly as descriptive approximations for nonsmooth maxima.
"""
    individual, joint = np.asarray(individual), np.asarray(joint)
    groups = np.asarray(groups)
    if (individual.ndim != 3 or joint.shape != individual.shape
            or groups.shape != (individual.shape[1],)):
        raise ValueError("Expected matching (context,item,target) losses and one group per item")
    if not np.isfinite(individual).all() or not np.isfinite(joint).all():
        raise ValueError("Losses must be finite")
    if repetitions < 100 or not 0 < alpha < 1:
        raise ValueError("Use at least 100 bootstrap draws and 0 < alpha < 1")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two independent base-item groups are required")
    # Sum per group, not mean: variable-size base items retain the original
    # pair-weighted estimand. A sampled base item contributes every pair.
    losses = np.stack([individual, joint])  # method, context, item, target
    group_sums = np.zeros((2, losses.shape[1], len(unique), losses.shape[3]))
    counts = np.bincount(inverse)
    for group in range(len(unique)):
        group_sums[:, :, group] = losses[:, :, inverse == group].sum(axis=2)
    point_means = losses.mean(axis=2)
    point_worst = point_means.max(axis=1)
    rng = np.random.default_rng(seed)
    all_worst = np.empty((repetitions, 2, individual.shape[2]))
    sup_deviation = np.empty(repetitions)
    for b in range(repetitions):
        draw = rng.integers(len(unique), size=len(unique))
        multiplicity = np.bincount(draw, minlength=len(unique))
        means = np.einsum("mcgr,g->mcr", group_sums, multiplicity) / (counts @ multiplicity)
        all_worst[b] = means.max(axis=1)
        sup_deviation[b] = np.abs(means - point_means).max()
    radius = float(np.quantile(sup_deviation, 1 - alpha))
    lower = np.maximum(point_means - radius, 0).max(axis=1)
    upper = np.minimum(point_means + radius, 1).max(axis=1)
    per_judge = []
    for j in range(individual.shape[2]):
        per_judge.append({
            "individual_worst_context_tv": float(point_worst[0, j]),
            "joint_worst_context_tv": float(point_worst[1, j]),
            "joint_minus_individual": float(point_worst[1, j] - point_worst[0, j]),
            "individual_simultaneous_interval": [float(lower[0, j]), float(upper[0, j])],
            "joint_simultaneous_interval": [float(lower[1, j]), float(upper[1, j])],
            "gap_simultaneous_interval": [float(lower[1, j] - upper[0, j]),
                                           float(upper[1, j] - lower[0, j])],
            "gap_paired_percentile_interval_descriptive": _interval(
                all_worst[:, 1, j] - all_worst[:, 0, j], alpha),
            "joint_lower_bound_above_gamma": bool(lower[1, j] > gamma),
        })
    panel = all_worst.max(axis=2)
    overall_lower, overall_upper = lower.max(axis=1), upper.max(axis=1)
    return {
        "unit": "base_item_id",
        "n_groups": len(unique),
        "n_pairs": len(groups),
        "repetitions": repetitions,
        "alpha": alpha,
        "seed": seed,
        "simultaneous_band_radius": radius,
        "per_removed_judge": per_judge,
        "overall": {
            "individual_max_tv": float(point_worst[0].max()),
            "joint_max_tv": float(point_worst[1].max()),
            "joint_minus_individual": float(point_worst[1].max() - point_worst[0].max()),
            "individual_simultaneous_interval": [float(overall_lower[0]), float(overall_upper[0])],
            "joint_simultaneous_interval": [float(overall_lower[1]), float(overall_upper[1])],
            "gap_simultaneous_interval": [float(overall_lower[1] - overall_upper[0]),
                                           float(overall_upper[1] - overall_lower[0])],
            "gap_paired_percentile_interval_descriptive": _interval(panel[:, 1] - panel[:, 0], alpha),
            "joint_lower_bound_above_gamma": bool(overall_lower[1] > gamma),
            "individual_upper_bound_at_most_gamma": bool(overall_upper[0] <= gamma),
            "positive_gap_by_simultaneous_band": bool(overall_lower[1] > overall_upper[0]),
            "noncomposability_supported_by_simultaneous_band": bool(
                overall_lower[1] > gamma and overall_upper[0] <= gamma),
        },
        "interpretation": (
            "Approximate cluster-bootstrap simultaneous confidence bands, conditional on the frozen "
            "FIT/CERT choice. Base-item groups must be independent. This is not a distribution-free "
            "certificate, and overlapping split seeds must not be counted as independent replicates."
        ),
    }


def run_experiment(data, gamma=.20, bootstrap=2000, alpha=.05, seed=0) -> dict:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1]")
    start = time.perf_counter()
    fit, cert, test = (data.splits[name] for name in ("FIT", "CERT", "TEST"))
    n = fit.n_judges
    W = loo_weights(fit)
    loo_cert = loss_cube(cert, range(n), W).mean(axis=1)
    removable = np.flatnonzero(loo_cert.max(axis=0) <= gamma).tolist()
    kept = sorted(set(range(n)) - set(removable))
    result = {
        "experiment": "exp01_dependency_significance",
        "gamma": gamma,
        "metadata": data.metadata,
        "judge_ids": list(data.judge_ids),
        "context_names": list(data.context_names),
        "selection_protocol": "LOO weights: FIT; removal decision: CERT; final reporting: untouched TEST",
        "removed_indices": removable,
        "removed_judge_ids": [data.judge_ids[j] for j in removable],
        "kept_indices": kept,
        "loo_weight_matrix": W.tolist(),
        "loo_cert_context_tv": loo_cert.tolist(),
        "loo_cert_worst_context_tv": loo_cert.max(axis=0).tolist(),
        "status": "complete",
        "caveats": [
            "This is a new held-out audit; it need not reproduce an original CERT-selected figure on CERT.",
            "The dependency graph belongs to the fitted certificates; nonunique LP solutions may yield other graphs.",
            "A loose or failed sufficient composition bound does not prove that any joint replacement must fail.",
            "The reported dependency bound is deterministic for each observed split, not a population-risk guarantee.",
        ],
    }
    if not removable:
        result.update(status="no_individually_removable_judges", elapsed_seconds=time.perf_counter() - start)
        return result
    dependency = dependency_composition(W, removable)
    result["dependency"] = dependency
    if not kept:
        result.update(status="all_judges_marked_removable_no_retained_endpoint",
                      elapsed_seconds=time.perf_counter() - start)
        return result
    joint_W = np.zeros((len(removable), n))
    for row, target in enumerate(removable):
        _, joint_W[row, kept] = solve_minimax_weights(fit, target, kept)
    individual_test = loss_cube(test, removable, W[removable])
    joint_test = loss_cube(test, removable, joint_W)
    result["joint_refit_weight_matrix"] = joint_W.tolist()
    result["test_individual_context_tv"] = individual_test.mean(axis=1).tolist()
    result["test_joint_context_tv"] = joint_test.mean(axis=1).tolist()
    result["uncertainty"] = paired_group_bootstrap(
        individual_test, joint_test, data.groups["TEST"], gamma, bootstrap, alpha, seed)
    for row, target in enumerate(removable):
        result["uncertainty"]["per_removed_judge"][row]["judge_id"] = data.judge_ids[target]
    if dependency["status"] == "contracting":
        inverse = np.asarray(dependency["inverse"])
        composed_W = np.zeros_like(joint_W)
        composed_W[:, kept] = np.asarray(dependency["composed_weights"])
        dependency["split_comparison"] = {}
        for split, responses in data.splits.items():
            eps = loss_cube(responses, removable, W[removable]).mean(axis=1)
            bound = eps @ inverse.T
            theorem_bound = inverse @ eps.max(axis=0)
            composed = loss_cube(responses, removable, composed_W).mean(axis=1)
            direct = loss_cube(responses, removable, joint_W).mean(axis=1)
            dependency["split_comparison"][split] = {
                "loo_residual_context_tv": eps.tolist(),
                "bound_context_tv_unclipped": bound.tolist(),
                "bound_worst_context_per_judge": bound.max(axis=0).tolist(),
                "bound_worst_context_clipped_at_one": np.minimum(bound.max(axis=0), 1).tolist(),
                "theorem2_robust_residual_bound_per_judge": theorem_bound.tolist(),
                "theorem2_sufficient_at_gamma_on_this_split": bool(theorem_bound.max() <= gamma),
                "composed_context_tv": composed.tolist(),
                "direct_refit_context_tv": direct.tolist(),
                "max_bound_violation_numerical": float(np.maximum(composed - bound, 0).max()),
                "composition_sufficient_at_gamma_on_this_split": bool(bound.max() <= gamma),
            }
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def write_figure(result: dict, destination: Path):
    """Paper-ready figure of a frozen decision; its title labels synthetic data."""
    if "uncertainty" not in result:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["uncertainty"]["per_removed_judge"]
    labels = result["removed_judge_ids"]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.5), 3.8))
    x = np.arange(len(rows))
    for shift, name, color, label in [(-.13, "individual", "#3366aa", "Individual removal"),
                                      (.13, "joint", "#c64545", "Joint refit")]:
        y = np.array([r[f"{name}_worst_context_tv"] for r in rows])
        intervals = np.array([r[f"{name}_simultaneous_interval"] for r in rows])
        axes[0].errorbar(x + shift, y, yerr=[y - intervals[:, 0], intervals[:, 1] - y],
                        fmt="o", color=color, capsize=3, label=label)
    dependency = result["dependency"]
    if dependency["status"] == "contracting":
        comp = dependency["split_comparison"]["TEST"]
        axes[0].scatter(x, np.asarray(comp["composed_context_tv"]).max(axis=0),
                        marker="x", color="#287a39", label="Composed witnesses")
        bound = np.minimum(comp["theorem2_robust_residual_bound_per_judge"], 1)
        axes[0].scatter(x, bound, marker="^", facecolors="none", edgecolors="#704699",
                        label="Theorem 2 bound (clipped at 1)")
    axes[0].axhline(result["gamma"], linestyle="--", color="black", label="Tolerance")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Held-out worst-context mean TV", ylim=(-.02, 1.04))
    axes[0].legend(fontsize=7)
    matrix = np.asarray(dependency["A_removed_to_removed"])
    shown = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1].set(xticks=x, xticklabels=labels, yticks=x, yticklabels=labels,
                xlabel="Removed donor", ylabel="Removed target",
                title=f"Dependency A\nspectral radius = {dependency['spectral_radius']:.3f}")
    fig.colorbar(shown, ax=axes[1], label="FIT replacement weight")
    metadata = result.get("metadata", {})
    synthetic = metadata.get("synthetic", metadata.get("is_demo", False))
    # Preserve the data provenance in the graphic even if the shared loader's
    # metadata schema changes; caller additionally sets source_mode below.
    if synthetic or result.get("source_mode") == "demo":
        fig.suptitle("SYNTHETIC DEMO — code verification only")
    else:
        fig.suptitle("Fixed FIT/CERT decision; TEST base-item bootstrap intervals")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=200)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main():
    from supplementary.common import add_data_arguments, load_dataset, save_json
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--gamma", type=float, default=.20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--plot", action="store_true", help="Also write PNG/PDF next to the JSON")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    data = load_dataset(args)
    result = run_experiment(data, args.gamma, args.bootstrap, args.alpha, args.seed)
    result["source_mode"] = "demo" if getattr(args, "demo", False) else "cache"
    save_json(args.output, result)
    if args.plot:
        write_figure(result, args.output.with_suffix(""))
    print(f"{result['status']}: {args.output}")


if __name__ == "__main__":
    main()
