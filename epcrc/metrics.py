"""Detailed per-step metrics for coverage-based pruning.

Given a CoverageFunctional and a kept set S, this module computes a rich dict
of metrics that both forward and backward experiments dump at each iteration.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from .coverage import CoverageFunctional


def step_metrics(
    coverage_fn: CoverageFunctional,
    kept_set: Set[int],
    gamma: float,
) -> Dict[str, object]:
    """Compute a rich per-step metrics dict for a kept set S.

    Returns a JSON-serializable dict keyed by metric name.
    """
    S = set(int(i) for i in kept_set)
    _, certs = coverage_fn.compute_coverage(S, return_certificates=True)
    assert certs is not None
    N = coverage_fn.N
    model_names = coverage_fn.model_names

    # Per-model uniqueness vector, ordered by model index
    U = np.array([certs[i].uniqueness for i in range(N)], dtype=float)

    # External (target not in S) uniqueness only
    ext_mask = np.array([i not in S for i in range(N)], dtype=bool)
    U_ext = U[ext_mask] if ext_mask.any() else np.array([0.0])

    # Routing-weight diagnostics over external targets:
    # average sparsity, max weight, entropy
    weight_sparsity: List[float] = []
    weight_max: List[float] = []
    weight_entropy: List[float] = []
    for i in range(N):
        if i in S:
            continue
        w = certs[i].weights
        if w.size == 0:
            continue
        w_pos = np.clip(w, 0.0, None)
        s = w_pos.sum()
        if s <= 0:
            continue
        w_pos = w_pos / s
        weight_sparsity.append(float((w_pos > 1e-6).sum()))
        weight_max.append(float(w_pos.max()))
        nz = w_pos[w_pos > 1e-12]
        ent = float(-np.sum(nz * np.log(nz))) if nz.size > 0 else 0.0
        weight_entropy.append(ent)

    coverage = float(np.max(U))
    sum_uniqueness = float(np.sum(U))
    kept_list = sorted(S)

    def _percentile(arr: np.ndarray, q: float) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, q))

    bottleneck_idx = int(np.argmax(U))
    metrics: Dict[str, object] = {
        "kept_set_idx": kept_list,
        "kept_set_names": [model_names[i] for i in kept_list],
        "kept_set_size": len(S),
        "removed_set_idx": [i for i in range(N) if i not in S],
        "removed_set_names": [model_names[i] for i in range(N) if i not in S],
        "removed_set_size": N - len(S),
        # Core coverage metrics
        "coverage_E_S": coverage,
        "sum_uniqueness_U": sum_uniqueness,
        "satisfies_gamma": bool(coverage <= gamma),
        # Summary stats of U(i|S) over all i
        "U_mean": float(np.mean(U)),
        "U_median": float(np.median(U)),
        "U_std": float(np.std(U)),
        "U_min": float(np.min(U)),
        "U_max": float(np.max(U)),
        "U_p25": _percentile(U, 25),
        "U_p75": _percentile(U, 75),
        "U_p90": _percentile(U, 90),
        "U_p95": _percentile(U, 95),
        # Summary stats restricted to external (i not in S)
        "U_ext_mean": float(np.mean(U_ext)),
        "U_ext_max": float(np.max(U_ext)),
        "U_ext_sum": float(np.sum(U_ext)),
        # How many models currently violate tolerance
        "n_models_violating_gamma": int(np.sum(U > gamma)),
        # Bottleneck (argmax U)
        "bottleneck_model_idx": bottleneck_idx,
        "bottleneck_model_name": model_names[bottleneck_idx],
        "bottleneck_uniqueness": float(U[bottleneck_idx]),
        # Routing diagnostics across external targets
        "routing_avg_n_nonzero_weights": (
            float(np.mean(weight_sparsity)) if weight_sparsity else 0.0
        ),
        "routing_max_weight_mean": (
            float(np.mean(weight_max)) if weight_max else 0.0
        ),
        "routing_weight_entropy_mean": (
            float(np.mean(weight_entropy)) if weight_entropy else 0.0
        ),
        # Per-model uniqueness dict
        "per_model_uniqueness": {model_names[i]: float(U[i]) for i in range(N)},
    }
    return metrics
