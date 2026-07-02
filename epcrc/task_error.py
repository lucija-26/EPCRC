"""Task-error preservation (the "beta" constraint).

The gamma constraint bounds *behavioral fidelity*: the router must mimic the
removed model's outputs.  This module measures *utility preservation*: after
substituting model i by its certificate router, how much worse does the system
predict the actual target (ground-truth flow in UTD19)?

For kept set S with certificates w_i(S):

    err_orig(i) = loss(Y_eval[:, i]        - y_true)      original model
    err_sub(i)  = loss(Y_eval[:, S] @ w_i  - y_true)      certificate router
    delta(i|S)  = err_sub(i) - err_orig(i)                task-error change

    B(S) = max_i delta(i|S)          (absolute form)
    B_rel(S) = max_i delta(i|S) / max(err_orig(i), eps)   (relative form)

A pruning run is beta-feasible when B(S) <= beta.  Notes:

  - delta can be NEGATIVE: a convex mixture can beat the original model on the
    task (variance reduction).  That is why the constraint should be one-sided
    ("do not degrade by more than beta"), not |delta| <= beta.
  - gamma already implies a bound here: by the triangle inequality,
    |err_sub - err_orig| <= loss(router - original) = U(i|S) <= gamma
    when `loss` is the same norm as the uniqueness metric.  A separate beta
    only adds information when loss differs from the fidelity metric (e.g.
    RMSE vs mean_abs), when beta << gamma, or in the relative form.
  - B(S) is NOT provably monotone in S (weights are fit for fidelity, not for
    task error), so backward/k-swap remain heuristics under a joint
    (gamma, beta) test -- the exhaustive and MILP optima stay well-defined.

Requires ground truth at the eval query points: regenerate the UTD19 bundle
(delete bundle.npz, rerun the pipeline) to populate `y_eval_true`.
"""
from __future__ import annotations

from typing import Callable, Dict, Set, Tuple, Union

import numpy as np

from .coverage import CoverageFunctional

LossFn = Callable[[np.ndarray], float]

_LOSSES: Dict[str, LossFn] = {
    "rmse": lambda r: float(np.sqrt(np.mean(r**2))),
    "mse": lambda r: float(np.mean(r**2)),
    "mean_abs": lambda r: float(np.mean(np.abs(r))),
}


def substitution_task_errors(
    cov: CoverageFunctional,
    kept_set: Set[int],
    y_eval_true: np.ndarray,
    loss: Union[str, LossFn] = "rmse",
) -> Dict[int, Tuple[float, float, float]]:
    """Per-model (err_orig, err_sub, delta) on the eval sample.

    Models inside S substitute themselves (delta = 0 by construction).
    """
    loss_fn = _LOSSES[loss] if isinstance(loss, str) else loss
    y_true = np.asarray(y_eval_true, dtype=float).reshape(-1)
    if y_true.shape[0] != cov.Y_eval.shape[0]:
        raise ValueError(
            f"y_eval_true has {y_true.shape[0]} rows, Y_eval has {cov.Y_eval.shape[0]}"
        )

    _, certs = cov.compute_coverage(kept_set, return_certificates=True)
    assert certs is not None
    Pe = cov.Y_eval[:, sorted(kept_set)]

    out: Dict[int, Tuple[float, float, float]] = {}
    for i in range(cov.N):
        pred_orig = cov.Y_eval[:, i]
        pred_sub = Pe @ certs[i].weights
        err_orig = loss_fn(pred_orig - y_true)
        err_sub = loss_fn(pred_sub - y_true)
        out[i] = (err_orig, err_sub, err_sub - err_orig)
    return out


def beta_coverage(
    cov: CoverageFunctional,
    kept_set: Set[int],
    y_eval_true: np.ndarray,
    loss: Union[str, LossFn] = "rmse",
    relative: bool = False,
    eps: float = 1e-12,
) -> float:
    """B(S): worst task-error degradation over all substituted models.

    relative=True returns max_i delta_i / err_orig_i (e.g. beta = 0.05 means
    "no model's task error may grow by more than 5% under substitution").
    """
    errs = substitution_task_errors(cov, kept_set, y_eval_true, loss=loss)
    if relative:
        return max(d / max(o, eps) for o, _, d in errs.values())
    return max(d for _, _, d in errs.values())


def quality_eligible_set(
    cov: CoverageFunctional,
    y_eval_true: np.ndarray,
    beta: float,
    loss: Union[str, LossFn] = "rmse",
) -> Set[int]:
    """Models whose OWN task error on the shared eval sample is <= beta.

    The quality-by-construction design: restrict the kept set S to this pool,
    then prune for coverage as usual.  Because task losses are convex and
    routing weights live on the simplex (Jensen, per query):

        L(sum_j w_j Y_j, y*) <= sum_j w_j L(Y_j, y*) <= max_{j in S} L(Y_j, y*)

    so EVERY certificate router over an eligible S automatically has task
    error <= beta -- no extra constraint inside the pruning loop, and the
    monotone coverage theory is untouched.  The two-tier problem becomes

        min |S|  s.t.  S ⊆ eligible,  max_{i in J} U(i|S) <= gamma,

    which may be INFEASIBLE if a hull archetype fails the quality bar; that
    outcome ("cannot prune to quality-beta representatives at tolerance
    gamma") is itself a reportable result.

    NOTE: the bound needs each model's error on the SHARED POOLED eval
    sample (all cities' queries), not its home-city training RMSE.
    """
    y_true = np.asarray(y_eval_true, dtype=float).reshape(-1)
    loss_fn = _LOSSES[loss] if isinstance(loss, str) else loss
    return {
        j for j in range(cov.N)
        if loss_fn(cov.Y_eval[:, j] - y_true) <= beta
    }


def joint_feasible(
    cov: CoverageFunctional,
    kept_set: Set[int],
    gamma: float,
    y_eval_true: np.ndarray,
    beta: float,
    loss: Union[str, LossFn] = "rmse",
    relative: bool = False,
) -> bool:
    """Feasibility under BOTH constraints: E(S) <= gamma and B(S) <= beta.

    Drop-in replacement for the `E <= gamma` test inside any pruner loop.
    """
    E, _ = cov.compute_coverage(kept_set)
    if E > gamma:
        return False
    return beta_coverage(cov, kept_set, y_eval_true, loss=loss, relative=relative) <= beta
