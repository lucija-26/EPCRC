"""Downstream preservation: does the compressed panel decide the same things?

Plan section 24 (experiment E2), claim C5.

Every other claim measures how far a reconstructed judge's *distribution*
moves.  That is the right object for a certificate, but it is not what anyone
deploys a panel to do.  C5 asks the operational question instead: aggregate the
panel into one decision per item, and check whether the aggregate built from
reconstructions matches the aggregate built from all twenty real judges.

The two can come apart in either direction, which is why this is a separate
claim rather than a corollary.  Reconstruction errors that are large but
cancel across judges leave the aggregate untouched; errors that are small but
aligned can still flip it.

Aggregators take the (n_items, n_judges, 3) block for one context and return
(n_items, 3).  Learned ones are fitted on FIT and applied unchanged elsewhere,
so nothing here ever sees the split it is scored on.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Smallest probability admitted before taking a log, so a confident aggregate
# that happens to be wrong contributes a large but finite NLL instead of inf.
_EPSILON = 1e-12


def mean_aggregate(block: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Uniform (or weighted) mean of the judges' distributions."""
    if weights is None:
        return block.mean(axis=1)
    w = np.asarray(weights, dtype=float)
    return np.tensordot(block, w / w.sum(), axes=([1], [0]))


def majority_aggregate(block: np.ndarray) -> np.ndarray:
    """Each judge casts one hard vote; the result is the vote share.

    Discards the judges' confidence entirely, which is exactly why it is worth
    reporting: if compression preserved only the argmax it would look perfect
    here and poor under the probabilistic metrics.
    """
    votes = block.argmax(axis=2)
    counts = np.stack([(votes == k).sum(axis=1) for k in range(3)], axis=1)
    return counts / counts.sum(axis=1, keepdims=True)


def fit_logistic_aggregate(
    block: np.ndarray, gold: np.ndarray, max_iter: int = 500
) -> np.ndarray:
    """Multinomial logistic calibration over the judges, trained on FIT.

    Features are the flattened judge distributions, so the aggregator can learn
    both which judges to trust and how to correct a systematic tie bias.
    """
    from sklearn.linear_model import LogisticRegression

    n_items = block.shape[0]
    model = LogisticRegression(max_iter=max_iter)
    model.fit(block.reshape(n_items, -1), gold)
    return model


def apply_logistic_aggregate(model, block: np.ndarray) -> np.ndarray:
    n_items = block.shape[0]
    probabilities = model.predict_proba(block.reshape(n_items, -1))
    # `classes_` may omit a label absent from FIT; re-expand to three columns.
    out = np.zeros((n_items, 3), dtype=float)
    for column, label in enumerate(model.classes_):
        out[:, int(label)] = probabilities[:, column]
    return out


# --------------------------------------------------------------------------
# item-level metrics
# --------------------------------------------------------------------------

def accuracy(aggregate: np.ndarray, gold: np.ndarray) -> float:
    return float((aggregate.argmax(axis=1) == gold).mean())


def macro_accuracy(aggregate: np.ndarray, gold: np.ndarray) -> float:
    """Mean per-class recall, so the 96 tie items are not drowned out."""
    predicted = aggregate.argmax(axis=1)
    recalls = [
        float((predicted[gold == k] == k).mean())
        for k in range(3)
        if (gold == k).any()
    ]
    return float(np.mean(recalls))


def negative_log_likelihood(aggregate: np.ndarray, gold: np.ndarray) -> float:
    picked = aggregate[np.arange(len(gold)), gold]
    return float(-np.log(np.clip(picked, _EPSILON, None)).mean())


def brier_score(aggregate: np.ndarray, gold: np.ndarray) -> float:
    onehot = np.zeros_like(aggregate)
    onehot[np.arange(len(gold)), gold] = 1.0
    return float(((aggregate - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(
    aggregate: np.ndarray, gold: np.ndarray, n_bins: int = 10
) -> float:
    """Gap between confidence and accuracy, averaged over equal-width bins."""
    confidence = aggregate.max(axis=1)
    correct = (aggregate.argmax(axis=1) == gold).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(total)


def disagreement_rate(block: np.ndarray) -> float:
    """Fraction of items where the judges do not all return the same verdict.

    A compressed panel that quietly collapses onto one opinion would keep its
    accuracy while losing the disagreement signal that makes a panel worth
    running, so this is reported next to the accuracy numbers.
    """
    votes = block.argmax(axis=2)
    return float((votes != votes[:, :1]).any(axis=1).mean())


# --------------------------------------------------------------------------
# agreement with the full panel
# --------------------------------------------------------------------------

def verdict_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.argmax(axis=1) == b.argmax(axis=1)).mean())


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau

    if len(a) < 2:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def preference_score(aggregate: np.ndarray) -> np.ndarray:
    """One scalar per item: how strongly the panel prefers A over B.

    Ties contribute to neither side, so this is the natural score to rank by
    and is what the ranking metrics below are computed on.
    """
    return aggregate[:, 0] - aggregate[:, 1]


def ranking_metrics(
    reference: np.ndarray, candidate: np.ndarray, groups: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Rank agreement between two aggregates.

    With `groups` the scores are first averaged within a group, which is how
    the domain-level ranking is produced.  Without it every item is its own
    unit.

    This stands in for the plan's system-level ranking.  RewardBench 2 does not
    record which system produced a response, so no true system axis exists in
    this data; that part of E2 needs the JuStRank source in E7.  The surrogate
    is labelled as such wherever it is reported and is not presented as a
    system ranking.
    """
    a = preference_score(reference)
    b = preference_score(candidate)

    if groups is not None:
        n_groups = int(groups.max()) + 1
        counts = np.bincount(groups, minlength=n_groups)
        a = np.bincount(groups, weights=a, minlength=n_groups) / counts
        b = np.bincount(groups, weights=b, minlength=n_groups) / counts

    order_a = np.argsort(np.argsort(-a))
    order_b = np.argsort(np.argsort(-b))

    return {
        "kendall_tau": kendall_tau(a, b),
        "spearman": spearman(a, b),
        "top1_agreement": float(np.argmax(a) == np.argmax(b)),
        "max_rank_displacement": float(np.abs(order_a - order_b).max()),
        "mean_rank_displacement": float(np.abs(order_a - order_b).mean()),
        "n_units": int(len(a)),
    }


def top_k_overlap(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    a = set(np.argsort(-preference_score(reference))[:k].tolist())
    b = set(np.argsort(-preference_score(candidate))[:k].tolist())
    return len(a & b) / k


def evaluate_aggregate(
    aggregate: np.ndarray,
    gold: np.ndarray,
    reference: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Every item-level metric, plus agreement with `reference` when given."""
    out = {
        "accuracy": accuracy(aggregate, gold),
        "macro_accuracy": macro_accuracy(aggregate, gold),
        "nll": negative_log_likelihood(aggregate, gold),
        "brier": brier_score(aggregate, gold),
        "ece": expected_calibration_error(aggregate, gold),
    }
    if reference is not None:
        out["verdict_agreement_vs_full"] = verdict_agreement(aggregate, reference)
        for name, value in ranking_metrics(reference, aggregate).items():
            out[f"item_rank_{name}"] = value
        if groups is not None:
            for name, value in ranking_metrics(reference, aggregate, groups).items():
                out[f"domain_rank_{name}"] = value
    return out
