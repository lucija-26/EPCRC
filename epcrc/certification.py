"""Honest certification of a compressed panel (plan section 27, experiment E5).

A compressed panel is only useful if the tolerance it promises survives contact
with items nobody fitted on.  Two things spoil that promise:

*reuse*
    weights chosen to minimise an error, scored on the same items that chose
    them, report an error biased downward;

*selection over a finite sample*
    even on fresh items the reported error is a mean over finitely many items,
    and a panel accepted because that mean landed below gamma is partly a panel
    that got lucky.

The rule this module implements fixes the first by fitting on FIT and measuring
on a disjoint CERT split, and the second by replacing the measured mean with a
one-sided upper confidence bound.  A panel is certified at tolerance gamma only
when

    max_{i, c} UCB_{1-delta}(U_{i,c})  <=  gamma

over every judge-context pair, so the correction is simultaneous rather than
per-pair.  `experiment_c7_certification.py` then checks, over many resplits,
how often a certified panel nonetheless exceeds gamma on the locked TEST split.

Losses are total variation and therefore live in [0, 1], which is what lets the
empirical Bernstein bound apply without any distributional assumption.

**Base items, not pairs.**  One base item yields several comparison pairs whose
losses are dependent, so a bound treating pairs as independent units would be
anticonservative for exactly the reason the module exists.  Every statistic
here first averages within a base item and then treats base items as the
independent sample.

What this does *not* give: validity uniform over the adaptive sequence of sets a
selection run visits.  The correction is over judge-context pairs at a fixed
candidate panel.  Selection happens on FIT and certification on CERT, so the
panel is fixed before CERT is touched, which is what makes the fixed-set bound
the right one here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import norm

from .judge import JudgeResponses, solve_minimax_weights, total_variation

# Total variation between three-class rows never exceeds 1.
LOSS_RANGE = 1.0


def per_item_losses(
    responses: JudgeResponses,
    target_idx: int,
    kept_list: Sequence[int],
    weights: np.ndarray,
) -> List[np.ndarray]:
    """Per-context arrays of the per-item TV loss of one reconstruction."""
    kept_list = list(kept_list)
    w = np.asarray(weights, dtype=float).reshape(-1)
    out = []
    for block in responses.blocks:
        recon = np.tensordot(block[:, kept_list, :], w, axes=([1], [0]))
        out.append(total_variation(block[:, target_idx, :], recon))
    return out


def group_mean(losses: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average `losses` within each base item, giving one value per base item."""
    n_groups = int(groups.max()) + 1
    totals = np.bincount(groups, weights=losses, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    return totals / counts


def bernstein_ucb(sample: np.ndarray, delta: float) -> float:
    """Maurer-Pontil empirical Bernstein upper bound on the mean.

    For `n` independent values in [0, `LOSS_RANGE`], with probability at least
    1 - delta the mean is at most

        xbar + sqrt(2 V ln(2/delta) / n) + 7 b ln(2/delta) / (3 (n - 1)),

    where V is the sample variance.  The variance term is what makes this worth
    using over Hoeffding: reconstruction losses are strongly concentrated for
    the judges that matter, so V is far below the worst case b^2 / 4 that
    Hoeffding must assume.
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 2:
        return float("inf")
    log_term = float(np.log(2.0 / delta))
    var = float(x.var(ddof=1))
    return float(
        x.mean()
        + np.sqrt(2.0 * var * log_term / n)
        + 7.0 * LOSS_RANGE * log_term / (3.0 * (n - 1))
    )


def normal_ucb(sample: np.ndarray, delta: float) -> float:
    """One-sided normal (CLT) bound, the cheap approximation Bernstein replaces.

    Reported as a comparison arm only: it has no finite-sample guarantee, and
    the point of E5 is to show what that costs.
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 2:
        return float("inf")
    z = float(norm.ppf(1.0 - delta))
    return float(x.mean() + z * x.std(ddof=1) / np.sqrt(n))


def fit_reconstructions(
    fit: JudgeResponses,
    kept_set: Set[int],
    n_judges: int,
) -> Dict[int, np.ndarray]:
    """Minimax weights on FIT for every judge outside `kept_set`."""
    kept_list = sorted(kept_set)
    return {
        target: solve_minimax_weights(fit, target, kept_list)[1]
        for target in range(n_judges)
        if target not in kept_set
    }


def pair_samples(
    cert: JudgeResponses,
    groups: np.ndarray,
    kept_set: Set[int],
    weights: Dict[int, np.ndarray],
) -> Dict[Tuple[int, int], np.ndarray]:
    """Base-item loss samples, one array per certified (judge, context) pair."""
    kept_list = sorted(kept_set)
    samples: Dict[Tuple[int, int], np.ndarray] = {}
    for target, w in weights.items():
        for c, losses in enumerate(per_item_losses(cert, target, kept_list, w)):
            samples[(target, c)] = group_mean(losses, groups)
    return samples


def certified_error(
    samples: Dict[Tuple[int, int], np.ndarray],
    delta: float,
    bound: str,
    correct: bool,
) -> float:
    """max over judge-context pairs of the chosen upper bound.

    `correct` applies the Bonferroni split of delta across the pairs, which is
    what makes the statement simultaneous.  Turning it off is the
    "uncorrected per-pair bound" comparison arm.
    """
    if not samples:
        return 0.0
    level = delta / len(samples) if correct else delta
    fn = {"bernstein": bernstein_ucb, "normal": normal_ucb}[bound]
    return max(fn(sample, level) for sample in samples.values())


def bootstrap_max_error(
    samples: Dict[Tuple[int, int], np.ndarray],
    delta: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """Grouped bootstrap bound on the maximum judge-context error.

    Resamples base items -- the same items for every pair, so the dependence
    between judges and between contexts is carried through rather than assumed
    away -- and takes the 1 - delta quantile of

        max_{i, c} ( mean_b(i, c) - mean_obs(i, c) ).

    Adding that one quantile to every observed mean bounds all pairs at once,
    so it is simultaneous like Bonferroni but pays for the real dependence
    instead of the worst case.
    """
    if not samples:
        return 0.0
    keys = sorted(samples)
    matrix = np.stack([samples[k] for k in keys])        # (n_pairs, n_items)
    observed = matrix.mean(axis=1)
    n_items = matrix.shape[1]

    rng = np.random.default_rng(seed)
    deviations = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        deviations[b] = float((matrix[:, idx].mean(axis=1) - observed).max())

    return float(observed.max() + np.quantile(deviations, 1.0 - delta))


def empirical_error(samples: Dict[Tuple[int, int], np.ndarray]) -> float:
    """max over pairs of the plain measured mean, with no allowance for noise."""
    if not samples:
        return 0.0
    return max(float(sample.mean()) for sample in samples.values())


def worst_test_error(
    test: JudgeResponses,
    kept_set: Set[int],
    weights: Dict[int, np.ndarray],
) -> float:
    """The quantity a certificate is a promise about: worst judge, worst context."""
    kept_list = sorted(kept_set)
    if not weights:
        return 0.0
    return max(
        float(max(losses.mean() for losses in per_item_losses(test, t, kept_list, w)))
        for t, w in weights.items()
    )
