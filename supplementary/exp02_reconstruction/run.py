"""Compare replacement maps on exactly the same, FIT-selected retained sets.

No removed endpoint's output is an inference feature. FIT trains coefficients,
CERT chooses temperatures/regularization, and TEST is evaluated once afterwards.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epcrc.judge import JudgeResponses, solve_minimax_weights, total_variation

METHODS = ("global_convex", "context_convex", "temperature_convex", "nonlinear_gate")


@dataclass
class TargetMap:
    weights: np.ndarray | None = None
    temperature: float = 1.0
    coef: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    def predict(self, retained: np.ndarray, context: int) -> np.ndarray:
        """Accept ONLY retained probabilities of shape (items, kept, classes)."""
        if self.coef is not None:
            raw = retained.reshape(len(retained), -1)
            features = np.column_stack((np.ones(len(raw)),
                                        (raw - self.feature_mean) / self.feature_scale))
            weights = softmax(features @ self.coef, axis=1)
            return np.einsum("nk,nkl->nl", weights, retained)
        weights = self.weights if self.weights.ndim == 1 else self.weights[context]
        mixture = np.einsum("nkl,k->nl", retained, weights)
        if self.temperature == 1.0:
            return mixture
        return softmax(np.log(np.clip(mixture, 1e-12, 1.0)) / self.temperature, axis=1)


def predict_all(responses: JudgeResponses, kept: list[int], models: dict[int, TargetMap]):
    """Retained interfaces are copied; all replacement features are restricted to S."""
    predictions = []
    for c, block in enumerate(responses.blocks):
        predicted = np.empty_like(block)
        predicted[:, kept, :] = block[:, kept, :]
        peers = block[:, kept, :]
        for target, model in models.items():
            predicted[:, target, :] = model.predict(peers, c)
        predictions.append(predicted)
    return predictions


def target_risk(responses, kept, target, model):
    return max(float(total_variation(block[:, target, :],
                                     model.predict(block[:, kept, :], c)).mean())
               for c, block in enumerate(responses.blocks))


def gate_objective(theta, features, peers, truth, context_slices, regularization,
                   epsilon=1e-3, beta=30.0):
    """Smooth minimax TV and its analytic derivative for an input-dependent gate."""
    coef = theta.reshape(features.shape[1], peers.shape[1])
    weights = softmax(features @ coef, axis=1)
    predicted = np.einsum("nk,nkl->nl", weights, peers)
    residual = predicted - truth
    smooth_abs = np.sqrt(residual * residual + epsilon * epsilon)
    losses = 0.5 * (smooth_abs - epsilon).sum(axis=1)
    context_means = np.asarray([losses[s].mean() for s in context_slices])
    risk = (logsumexp(beta * context_means) - np.log(len(context_slices))) / beta
    context_weights = softmax(beta * context_means)
    row_weights = np.zeros(len(features))
    for cw, selection in zip(context_weights, context_slices):
        row_weights[selection] = cw / len(losses[selection])
    dq = 0.5 * residual / smooth_abs * row_weights[:, None]
    dw = np.einsum("nl,nkl->nk", dq, peers)
    dz = weights * (dw - (dw * weights).sum(axis=1, keepdims=True))
    gradient = features.T @ dz
    denominator = max(coef[1:].size, 1)
    risk += regularization * np.square(coef[1:]).sum() / denominator
    gradient[1:] += 2.0 * regularization * coef[1:] / denominator
    return float(risk), gradient.ravel()


def fit_gate(fit, kept, target, initial_weights, regularization, maxiter):
    peers = np.concatenate([b[:, kept, :] for b in fit.blocks])
    truth = np.concatenate([b[:, target, :] for b in fit.blocks])
    raw = peers.reshape(len(peers), -1)
    mean, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-5)
    features = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    boundaries = np.cumsum([0] + [len(b) for b in fit.blocks])
    slices = [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    coef = np.zeros((features.shape[1], len(kept)))
    coef[0] = np.log(np.maximum(initial_weights, 1e-4))
    result = minimize(gate_objective, coef.ravel(),
                      args=(features, peers, truth, slices, regularization),
                      method="L-BFGS-B", jac=True,
                      options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError("Nonlinear gate returned non-finite parameters/loss")
    model = TargetMap(coef=result.x.reshape(coef.shape), feature_mean=mean, feature_scale=scale)
    diagnostics = {"regularization": regularization, "converged": bool(result.success),
                   "message": str(result.message), "iterations": int(result.nit),
                   "smooth_fit_objective": float(result.fun)}
    return model, diagnostics


def fit_reconstructors(fit, cert, kept, regularizations=(0.001, 0.01, 0.1),
                       temperatures=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0), maxiter=150):
    if not kept or len(set(kept)) != len(kept):
        raise ValueError("kept must contain distinct endpoints and cannot be empty")
    if any(j < 0 or j >= fit.n_judges for j in kept):
        raise ValueError("kept endpoint out of range")
    if not regularizations or min(regularizations) < 0 or not temperatures or min(temperatures) <= 0:
        raise ValueError("Require nonnegative regularization and positive temperatures")
    maps = {name: {} for name in METHODS}
    diagnostics = {name: {} for name in METHODS}
    timings = {name: 0.0 for name in METHODS}
    for target in sorted(set(range(fit.n_judges)) - set(kept)):
        started = time.perf_counter()
        fit_error, weights = solve_minimax_weights(fit, target, kept)
        maps["global_convex"][target] = TargetMap(weights=weights)
        diagnostics["global_convex"][target] = {"fit_worst_context_tv": float(fit_error)}
        timings["global_convex"] += time.perf_counter() - started

        started = time.perf_counter()
        per_context = [solve_minimax_weights(JudgeResponses([block]), target, kept)[1]
                       for block in fit.blocks]
        maps["context_convex"][target] = TargetMap(weights=np.stack(per_context))
        timings["context_convex"] += time.perf_counter() - started

        started = time.perf_counter()
        candidates = [TargetMap(weights=weights, temperature=float(t)) for t in temperatures]
        errors = [target_risk(cert, kept, target, model) for model in candidates]
        best = int(np.argmin(errors))
        maps["temperature_convex"][target] = candidates[best]
        diagnostics["temperature_convex"][target] = {
            "temperature": candidates[best].temperature,
            "cert_grid": [{"temperature": float(t), "worst_context_tv": float(e)}
                          for t, e in zip(temperatures, errors)]}
        timings["temperature_convex"] += time.perf_counter() - started

        started = time.perf_counter()
        candidates, trials = [], []
        for regularization in regularizations:
            model, diagnostic = fit_gate(fit, kept, target, weights, regularization, maxiter)
            diagnostic["cert_worst_context_tv"] = target_risk(cert, kept, target, model)
            candidates.append(model)
            trials.append(diagnostic)
        best = int(np.argmin([trial["cert_worst_context_tv"] for trial in trials]))
        maps["nonlinear_gate"][target] = candidates[best]
        diagnostics["nonlinear_gate"][target] = {"selected_trial": best, "trials": trials}
        timings["nonlinear_gate"] += time.perf_counter() - started
    # Temperature and gate training reuse global coefficients, so include that cost.
    timings["temperature_convex"] += timings["global_convex"]
    timings["nonlinear_gate"] += timings["global_convex"]
    return maps, diagnostics, timings


def evaluate(responses, predictions, kept):
    losses = [total_variation(block, prediction)
              for block, prediction in zip(responses.blocks, predictions)]
    context_endpoint = np.stack([loss.mean(axis=0) for loss in losses])
    removed = sorted(set(range(responses.n_judges)) - set(kept))
    verdict_correct = [b.argmax(axis=-1) == p.argmax(axis=-1)
                       for b, p in zip(responses.blocks, predictions)]
    invalid = [(~np.isfinite(p).all(axis=-1)) | (p.min(axis=-1) < -1e-7)
               | (np.abs(p.sum(axis=-1) - 1) > 1e-6) for p in predictions]
    metrics = {"worst_endpoint_context_tv": float(context_endpoint.max()),
               "mean_endpoint_context_tv": float(context_endpoint.mean()),
               "per_context_endpoint_mean_tv": context_endpoint.tolist(),
               "removed_verdict_agreement": (float(np.concatenate(verdict_correct)[:, removed].mean())
                                             if removed else 1.0),
               "all_endpoint_verdict_agreement": float(np.concatenate(verdict_correct).mean()),
               "off_simplex_fraction": float(np.concatenate(invalid).mean())}
    return metrics, losses


def bootstrap_difference(reference_losses, candidate_losses, groups, n_bootstrap, seed):
    """Paired cluster bootstrap, recomputing the max across endpoints and contexts."""
    if n_bootstrap == 0:
        return None
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), n_bootstrap)
    denominator = multiplicities @ counts
    risks = []
    for losses in (reference_losses, candidate_losses):
        context_risks = []
        for loss in losses:
            if len(loss) != len(groups):
                raise ValueError("Contexts and base item groups must have identical row alignment")
            cluster_sums = np.zeros((len(unique), loss.shape[1]))
            np.add.at(cluster_sums, inverse, loss)
            estimates = (multiplicities @ cluster_sums) / denominator[:, None]
            context_risks.append(estimates.max(axis=1))
        risks.append(np.max(context_risks, axis=0))
    differences = risks[1] - risks[0]
    return {"statistic": "candidate minus global convex worst endpoint/context TV",
            "percentile_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
            "n_base_items": len(unique), "replicates": n_bootstrap,
            "conditioning": "FIT/CERT models and subset frozen; TEST base items resampled jointly across contexts"}


def serialize_map(model):
    return {name: (value.tolist() if isinstance(value, np.ndarray) else value)
            for name, value in vars(model).items()}


def run(args):
    from supplementary.common import load_dataset, save_json, select_subset
    data = load_dataset(args)
    fit, cert, test = (data.splits[s] for s in ("FIT", "CERT", "TEST"))
    budgets = sorted(set(k for k in args.budgets if 1 <= k <= fit.n_judges))
    if not budgets:
        raise ValueError("No requested budget is valid for this ecosystem")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"experiment": "exp02_reconstruction", "metadata": data.metadata,
              "judge_ids": data.judge_ids, "contexts": data.context_names,
              "protocol": "Shared FIT-selected support; FIT coefficient training; CERT hyperparameters; untouched TEST",
              "frontier_scope": "Observed risk along shared supports, not the minimum size attainable by each map class",
              "selection_method": args.selection, "seed": args.seed,
              "configuration": {"regularizations": args.regularizations, "temperatures": args.temperatures,
                                "maxiter": args.maxiter, "bootstrap": args.bootstrap,
                                "gate_smooth_tv_epsilon": 1e-3, "gate_context_softmax_beta": 30.0},
              "gammas": args.gammas, "budgets": []}
    for k in budgets:
        started = time.perf_counter()
        kept = select_subset(fit, k, method="qr" if args.selection == "pivoted_qr" else args.selection,
                             seed=args.seed)
        selection_seconds = time.perf_counter() - started
        maps, diagnostics, training_seconds = fit_reconstructors(
            fit, cert, kept, tuple(args.regularizations), tuple(args.temperatures), args.maxiter)
        row = {"k": k, "retained_indices": kept,
               "retained_judge_ids": [data.judge_ids[j] for j in kept],
               "selection_seconds": selection_seconds, "methods": {}}
        test_losses, arrays = {}, {"test_base_item_groups": np.asarray(data.groups["TEST"]).astype(str)}
        parameters = {}
        for method in METHODS:
            started = time.perf_counter()
            predictions = predict_all(test, kept, maps[method])
            prediction_seconds = time.perf_counter() - started
            metrics, losses = evaluate(test, predictions, kept)
            cert_metrics, _ = evaluate(cert, predict_all(cert, kept, maps[method]), kept)
            fit_metrics, _ = evaluate(fit, predict_all(fit, kept, maps[method]), kept)
            test_losses[method] = losses
            for c, loss in enumerate(losses):
                arrays[f"{method}_tv_context_{c}"] = loss
            parameters[method] = {str(j): serialize_map(model) for j, model in maps[method].items()}
            row["methods"][method] = {"test": metrics, "cert": cert_metrics, "fit": fit_metrics,
                                       "training_seconds": training_seconds[method],
                                       "test_prediction_seconds_cpu": prediction_seconds,
                                       "fit_diagnostics": diagnostics[method]}
        for method in METHODS:
            row["methods"][method]["paired_difference_vs_global"] = bootstrap_difference(
                test_losses["global_convex"], test_losses[method], data.groups["TEST"],
                args.bootstrap, args.seed)
        loss_file, map_file = output / f"k{k}_test_losses.npz", output / f"k{k}_maps.json"
        np.savez_compressed(loss_file, **arrays)
        save_json(map_file, parameters)
        row.update({"test_losses_file": loss_file.name, "fitted_maps_file": map_file.name})
        result["budgets"].append(row)
        # Checkpoint makes slow full-panel runs inspectable without consuming TEST to adapt.
        save_json(output / "results.json", result)
        print(f"k={k}: " + ", ".join(f"{m}={row['methods'][m]['test']['worst_endpoint_context_tv']:.4f}"
                                      for m in METHODS), flush=True)
    result["cert_selected_budget_at_gamma"] = []
    result["observed_fit_budget_at_gamma"] = []
    for gamma in args.gammas:
        for method in METHODS:
            fitted_eligible = [r["k"] for r in result["budgets"]
                               if r["methods"][method]["fit"]["worst_endpoint_context_tv"] <= gamma]
            result["observed_fit_budget_at_gamma"].append({
                "gamma": gamma, "method": method,
                "observed_k": min(fitted_eligible) if fitted_eligible else None,
                "note": "Descriptive FIT risk at CERT-selected hyperparameters; shared supports, not a certified minimum"})
            eligible = [r for r in result["budgets"]
                        if r["methods"][method]["cert"]["worst_endpoint_context_tv"] <= gamma]
            selected = min(eligible, key=lambda r: r["k"]) if eligible else None
            result["cert_selected_budget_at_gamma"].append({
                "gamma": gamma, "method": method, "selected_k": selected["k"] if selected else None,
                "test_worst_endpoint_context_tv": (selected["methods"][method]["test"]["worst_endpoint_context_tv"]
                                                    if selected else None),
                "note": "CERT selection over declared tested budgets only; not an exact minimum or a confidence certificate"})
    save_json(output / "results.json", result)
    return result


def main():
    from supplementary.common import add_data_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 10, 12, 16, 18, 19, 20])
    parser.add_argument("--selection", choices=["backward", "pivoted_qr"], default="backward")
    parser.add_argument("--regularizations", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.75, 1., 1.25, 1.5, 2.])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.12, 0.20])
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", default="results/supplementary/exp02_reconstruction")
    args = parser.parse_args()
    if args.maxiter < 1 or args.bootstrap < 0:
        parser.error("maxiter must be positive and bootstrap nonnegative")
    run(args)


if __name__ == "__main__":
    main()
