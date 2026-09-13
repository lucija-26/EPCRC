"""Pairwise-geometry selection baselines (plan section 21.2).

These are the rules C3 has to beat.  Every one of them picks the physical
panel using only *pairwise* structure between judges -- a distance matrix, a
correlation matrix, or the column geometry of the response matrix.  None of
them can see what a convex combination of several judges would do, which is
exactly the gap C3 claims to exploit.

All selectors share one signature so the C3 driver can treat them uniformly::

    selector(panel, split="FIT") -> {k: [judge indices]}

and all of them look only at FIT, never at CERT or TEST.

The last entry, `lowrank_floor`, is deliberately *not* a selector.  Section
21.2 asks for a PCA/NMF reconstruction as a nondeployable lower bound: it may
use any rank-k subspace, not just one spanned by k actual judges, so it is a
floor on what any selection could achieve rather than a competitor.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import qr
from scipy.spatial.distance import squareform

from epcrc.judge import JudgeResponses, total_variation

__all__ = [
    "judge_matrix",
    "pairwise_tv",
    "select_correlation_medoid",
    "select_kmedoids",
    "select_farthest_first",
    "select_hierarchical",
    "select_pivoted_qr",
    "select_leverage",
    "lowrank_floor",
    "GEOMETRY_SELECTORS",
]


# --------------------------------------------------------------------------
# judge representations
# --------------------------------------------------------------------------

def judge_matrix(responses: JudgeResponses) -> np.ndarray:
    """Stack every judge into one long response vector.

    Returns shape ``(n_judges, n_rows)`` where a row is one (context, item,
    outcome) coordinate.  This is the "response vector" the plan refers to in
    21.2, and it is the representation the QR and low-rank baselines need.
    """
    per_context = [
        block.transpose(1, 0, 2).reshape(block.shape[1], -1)
        for block in responses.blocks
    ]
    return np.concatenate(per_context, axis=1)


def pairwise_tv(responses: JudgeResponses) -> np.ndarray:
    """Mean total-variation distance between every pair of judges.

    The mean is taken inside a context first and then across contexts, so a
    large clean split cannot dominate a small stress split -- the same
    weighting the coverage objective uses.
    """
    n = responses.n_judges
    per_context = np.empty((responses.n_contexts, n, n), dtype=float)

    for c, block in enumerate(responses.blocks):
        for a in range(n):
            per_context[c, a, a] = 0.0
            for b in range(a + 1, n):
                d = float(total_variation(block[:, a, :], block[:, b, :]).mean())
                per_context[c, a, b] = per_context[c, b, a] = d

    return per_context.mean(axis=0)


def _medoid(distance: np.ndarray, members: Sequence[int]) -> int:
    """Member with the smallest total distance to the rest of its cluster."""
    members = [int(m) for m in members]
    sub = distance[np.ix_(members, members)]
    return members[int(sub.sum(axis=1).argmin())]


def _chain_from_labels(
    distance: np.ndarray,
    labels_for_k,
    n: int,
) -> Dict[int, List[int]]:
    """Turn a clustering-at-every-k rule into the standard chain dictionary.

    `fcluster` can return fewer than k clusters when the linkage has tied
    merge heights, which would hand this baseline a smaller panel than its
    competitors and silently corrupt the equal-k comparison.  Short selections
    are therefore topped up by farthest-first, so every k really has k judges.
    """
    chain: Dict[int, List[int]] = {}
    for k in range(1, n + 1):
        labels = labels_for_k(k)
        picks = sorted({
            _medoid(distance, np.flatnonzero(labels == lab))
            for lab in np.unique(labels)
        })
        while len(picks) < k:
            covered = distance[:, picks].min(axis=1)
            covered[picks] = -np.inf
            picks = sorted(picks + [int(covered.argmax())])
        chain[k] = picks
    return chain


# --------------------------------------------------------------------------
# 21.2 baselines
# --------------------------------------------------------------------------

def select_correlation_medoid(
    responses: JudgeResponses,
    **_: object,
) -> Dict[int, List[int]]:
    """Correlation clustering followed by medoid selection.

    Judges are clustered by ``1 - |corr|`` on their response vectors, which is
    the standard "these two judges say the same thing" rule, and each cluster
    sends its medoid to the physical panel.
    """
    X = judge_matrix(responses)
    n = X.shape[0]

    corr = np.corrcoef(X)
    corr = np.nan_to_num(corr, nan=0.0)
    dissim = 1.0 - np.abs(corr)
    np.fill_diagonal(dissim, 0.0)
    dissim = np.clip((dissim + dissim.T) / 2.0, 0.0, None)

    Z = linkage(squareform(dissim, checks=False), method="average")
    return _chain_from_labels(
        dissim, lambda k: fcluster(Z, t=k, criterion="maxclust"), n
    )


def select_kmedoids(
    responses: JudgeResponses,
    n_restarts: int = 25,
    seed: int = 0,
    **_: object,
) -> Dict[int, List[int]]:
    """k-medoids on response vectors under the TV metric.

    Plain alternating PAM with random restarts.  The panel is small enough
    that restarts make this effectively exact, so the baseline is not being
    handicapped by a bad local optimum.
    """
    distance = pairwise_tv(responses)
    n = distance.shape[0]
    rng = np.random.default_rng(seed)

    chain: Dict[int, List[int]] = {}
    for k in range(1, n + 1):
        best, best_cost = None, np.inf
        for _restart in range(n_restarts):
            medoids = list(rng.choice(n, size=k, replace=False))
            for _sweep in range(100):
                assign = distance[:, medoids].argmin(axis=1)
                moved = []
                for c in range(k):
                    members = np.flatnonzero(assign == c)
                    if members.size == 0:
                        moved.append(medoids[c])
                    else:
                        moved.append(_medoid(distance, members))
                if sorted(moved) == sorted(medoids):
                    break
                medoids = moved
            cost = float(distance[:, medoids].min(axis=1).sum())
            if cost < best_cost:
                best, best_cost = sorted({int(m) for m in medoids}), cost
        chain[k] = best
    return chain


def select_farthest_first(
    responses: JudgeResponses,
    **_: object,
) -> Dict[int, List[int]]:
    """Farthest-first traversal, the classic k-center coreset heuristic.

    Seeded deterministically at the judge furthest from the panel centroid so
    the chain does not depend on an arbitrary starting choice.
    """
    distance = pairwise_tv(responses)
    n = distance.shape[0]

    order = [int(distance.sum(axis=1).argmax())]
    while len(order) < n:
        covered = distance[:, order].min(axis=1)
        covered[order] = -np.inf
        order.append(int(covered.argmax()))

    return {k: sorted(order[:k]) for k in range(1, n + 1)}


def select_hierarchical(
    responses: JudgeResponses,
    method: str = "average",
    **_: object,
) -> Dict[int, List[int]]:
    """Hierarchical clustering on the TV metric, one representative per cluster."""
    distance = pairwise_tv(responses)
    n = distance.shape[0]
    Z = linkage(squareform(distance, checks=False), method=method)
    return _chain_from_labels(
        distance, lambda k: fcluster(Z, t=k, criterion="maxclust"), n
    )


def select_pivoted_qr(
    responses: JudgeResponses,
    **_: object,
) -> Dict[int, List[int]]:
    """Column-pivoted QR on the response matrix.

    Judges are the columns; QR with column pivoting greedily takes the column
    least explained by those already chosen, which is the standard numerical
    answer to "pick k representative columns".  The pivot order is nested, so
    one factorisation gives the whole chain.
    """
    X = judge_matrix(responses).T          # (n_rows, n_judges)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, piv = qr(X, mode="economic", pivoting=True)
    order = [int(p) for p in piv]
    return {k: sorted(order[:k]) for k in range(1, X.shape[1] + 1)}


def select_leverage(
    responses: JudgeResponses,
    rank: int = None,
    **_: object,
) -> Dict[int, List[int]]:
    """Leverage-score column selection.

    Scores each judge by its statistical leverage in the top-`rank` right
    singular subspace and keeps the highest scorers.  Unlike pivoted QR this
    is a one-shot score rather than a greedy residual rule, so the two can
    disagree.
    """
    X = judge_matrix(responses).T
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[1]
    rank = min(n, X.shape[0]) if rank is None else rank

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    scores = (Vt[:rank] ** 2).sum(axis=0)
    order = [int(i) for i in np.argsort(scores)[::-1]]
    return {k: sorted(order[:k]) for k in range(1, n + 1)}


# --------------------------------------------------------------------------
# nondeployable floor
# --------------------------------------------------------------------------

def lowrank_floor(
    fit: JudgeResponses,
    evaluate: JudgeResponses,
    kind: str = "pca",
) -> Dict[int, Dict[str, float]]:
    """Rank-k reconstruction error using any subspace, not just real judges.

    This is the section 21.2 nondeployable bound.  The response matrix has one
    row per (context, item, outcome) and one column per judge, so a rank-k
    basis is a k-dimensional subspace of *judge space*.  That is what makes the
    bound transfer across splits: the basis is fitted on FIT and then applied
    to held-out rows, because it lives in judge coordinates rather than in item
    coordinates.

    It is a floor and not a competitor for two reasons.  It may use latent
    directions that no single judge occupies, and reading off a latent
    coordinate on a new item requires querying every judge -- which is the
    opposite of compression.  Its outputs are also not constrained to the
    simplex.  No panel of k physical judges can beat it.

    Returns ``{k: {"worst_judge_worst_context_tv": ..., "mean_...": ...}}``.
    """
    if kind not in ("pca", "nmf"):
        raise ValueError(f"kind must be 'pca' or 'nmf', got {kind!r}")

    X = judge_matrix(fit).T                 # (n_fit_rows, n_judges)
    Y = judge_matrix(evaluate).T            # (n_eval_rows, n_judges)
    n = X.shape[1]
    out: Dict[int, Dict[str, float]] = {}

    for k in range(1, n + 1):
        if kind == "pca":
            centre = X.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(X - centre, full_matrices=False)
            basis = Vt[:k]                                  # (k, n_judges)
            recon = (Y - centre) @ basis.T @ basis + centre
        else:
            from sklearn.decomposition import NMF

            model = NMF(n_components=k, init="nndsvda", max_iter=600,
                        random_state=0)
            model.fit(np.clip(X, 0.0, None))
            recon = model.transform(np.clip(Y, 0.0, None)) @ model.components_

        worst_per_judge = []
        for j in range(n):
            errs, offset = [], 0
            for block in evaluate.blocks:
                width = block.shape[0] * 3
                approx = recon[offset:offset + width, j].reshape(-1, 3)
                errs.append(float(
                    total_variation(block[:, j, :], approx).mean()
                ))
                offset += width
            worst_per_judge.append(max(errs))

        out[k] = {
            "worst_judge_worst_context_tv": float(max(worst_per_judge)),
            "mean_judge_worst_context_tv": float(np.mean(worst_per_judge)),
        }
    return out


GEOMETRY_SELECTORS = {
    "correlation_medoid": select_correlation_medoid,
    "kmedoids": select_kmedoids,
    "farthest_first": select_farthest_first,
    "hierarchical": select_hierarchical,
    "pivoted_qr": select_pivoted_qr,
    "leverage": select_leverage,
}
