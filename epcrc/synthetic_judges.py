"""Controlled judge panels whose correct answer is known before running anything.

Both constructions live in the probability simplex, which for three outcomes is
an equilateral triangle, so the panel geometry can be designed in the plane and
mapped to distributions by barycentric coordinates.  That map is affine, so a
judge built as a convex combination of others in the plane stays an exact
convex combination of them as a distribution.

Per-item variation is a contraction toward a random interior anchor,
q -> a + s (q - a) with s in (0, 1].  A contraction toward an interior point
maps the triangle into itself, so every generated row is a valid distribution
with no clipping, and being affine it leaves all convexity relations intact.

  * `duplicated_extremes`: every judge is individually redundant (its twin
    covers it exactly) yet one representative of each extreme must remain, so
    the naive simultaneous removal leaves nothing and the composition gap is 3.
  * `curved_arc`: judges evenly spaced on a circular arc.  A judge's neighbours
    miss it by the sagitta of the chord, which shrinks like k^-2 in the number
    of judges, giving a smooth family of near-redundant panels.

`archetype_panel` is the E8 generator and does not live in the plane.  A judge
there is a whole response *function* over items, so the judge cloud is a point
set in ``R^(n_items * 3)`` rather than in the triangle, and a planted backbone
of any size is identifiable.  See its docstring for why the backbone it plants
is the unique minimum.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from .judge import JudgeResponses

# Triangle whose corners are the pure outcomes (A better), (B better), (tie).
_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
_HEIGHT = np.sqrt(3.0) / 2.0
_CENTROID = _CORNERS.mean(axis=0)
# Distance from the centroid to an edge; a circle of this radius stays inside.
_INRADIUS = 1.0 / (2.0 * np.sqrt(3.0))


def to_simplex(points: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of planar points, i.e. their distributions."""
    points = np.asarray(points, dtype=float)
    p_tie = points[..., 1] / _HEIGHT
    p_b = points[..., 0] - 0.5 * p_tie
    p_a = 1.0 - p_b - p_tie
    return np.stack([p_a, p_b, p_tie], axis=-1)


def _random_interior(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform points inside the triangle, via Dirichlet barycentric weights."""
    return rng.dirichlet(np.ones(3), size=n) @ _CORNERS


def _render(
    latent: np.ndarray,
    n_items: int,
    rng: np.random.Generator,
    shrink: Tuple[float, float] = (0.55, 1.0),
) -> np.ndarray:
    """Turn planar judge positions into an (n_items, n_judges, 3) block."""
    anchors = _random_interior(rng, n_items)                 # (n_items, 2)
    scales = rng.uniform(shrink[0], shrink[1], size=n_items)  # (n_items,)
    placed = anchors[:, None, :] + scales[:, None, None] * (
        latent[None, :, :] - anchors[:, None, :]
    )
    return to_simplex(placed)


def _panel(
    latent: np.ndarray,
    n_items: int,
    n_contexts: int,
    seed: int,
    names: Optional[list] = None,
) -> Tuple[JudgeResponses, JudgeResponses, list]:
    """Independent fit and eval draws of the same planar configuration."""
    rng = np.random.default_rng(seed)
    context_names = ["clean"] + [f"stress_{c}" for c in range(1, n_contexts)]

    fit = JudgeResponses(
        [_render(latent, n_items, rng) for _ in range(n_contexts)], context_names
    )
    eval_ = JudgeResponses(
        [_render(latent, n_items, rng) for _ in range(n_contexts)], context_names
    )
    if names is None:
        names = [f"judge_{i}" for i in range(latent.shape[0])]
    return fit, eval_, names


def duplicated_extremes(
    n_items: int = 120,
    n_contexts: int = 2,
    inset: float = 0.8,
    jitter: float = 0.0,
    seed: int = 0,
) -> Tuple[JudgeResponses, JudgeResponses, list]:
    """Three extremes, each present twice: individually redundant, jointly not.

    With `jitter = 0` each judge is reproduced exactly by its twin, so every
    leave-one-out error is 0 and every judge looks removable.  Removing all six
    at once leaves an empty panel, while any feasible panel needs one judge from
    each of the three extremes.
    """
    base = _CENTROID + inset * (_CORNERS - _CENTROID)
    latent = np.repeat(base, 2, axis=0)
    if jitter > 0:
        latent = latent + np.random.default_rng(seed + 1).normal(
            scale=jitter, size=latent.shape
        )
    names = [f"{corner}_{copy}" for corner in ("A", "B", "tie") for copy in (1, 2)]
    return _panel(latent, n_items, n_contexts, seed, names)


def curved_arc(
    n_judges: int = 9,
    n_items: int = 120,
    radius: float = 0.20,
    arc_span: float = 1.6,
    n_contexts: int = 2,
    seed: int = 0,
) -> Tuple[JudgeResponses, JudgeResponses, list]:
    """Judges evenly spaced on a circular arc around the centre of the simplex.

    Interior judges sit just outside the chord joining their neighbours, missing
    it by the sagitta ``radius * (1 - cos(dtheta))`` for spacing ``dtheta``.
    Halving the spacing quarters that gap, which is the k^-2 law the curvature
    argument predicts.  The two endpoints are genuine extremes and can never be
    reconstructed.
    """
    if n_judges < 3:
        raise ValueError("curved_arc needs at least 3 judges")
    if radius >= _INRADIUS:
        raise ValueError(
            f"radius {radius} would leave the simplex; keep it below {_INRADIUS:.3f}"
        )

    angles = np.linspace(-arc_span / 2.0, arc_span / 2.0, n_judges) + np.pi / 2.0
    latent = _CENTROID + radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1
    )
    names = [f"arc_{i}" for i in range(n_judges)]
    return _panel(latent, n_items, n_contexts, seed, names)


def sagitta(radius: float, arc_span: float, n_judges: int) -> float:
    """Planar gap between an interior arc judge and the chord of its neighbours."""
    dtheta = arc_span / (n_judges - 1)
    return float(radius * (1.0 - np.cos(dtheta)))


class ArchetypePanel(NamedTuple):
    """A synthetic panel together with the backbone that generated it.

    `splits` holds independent FIT/CERT/TEST draws of the same judge
    configuration -- same archetypes, same mixing weights, fresh items -- so a
    tolerance fitted on one split can be honestly tested on another.
    `backbone` is the ground truth: the judge indices that are archetypes.
    """
    splits: dict
    names: list
    backbone: List[int]
    mixing: np.ndarray
    duplicate_of: dict


def _dirichlet_rows(rng: np.random.Generator, shape: Tuple[int, ...],
                    alpha: float) -> np.ndarray:
    """Independent Dirichlet draws with a common concentration."""
    return rng.dirichlet(np.full(3, alpha), size=shape)


def archetype_panel(
    n_judges: int = 12,
    n_archetypes: int = 4,
    n_items: int = 200,
    n_contexts: int = 2,
    noise: float = 0.0,
    noise_correlation: float = 0.5,
    n_duplicates: int = 0,
    concentration: float = 0.35,
    seed: int = 0,
) -> ArchetypePanel:
    """A panel built from a known backbone of `n_archetypes` archetypal judges.

    Construction, per context, with the mixing weights shared across contexts
    because they are what makes a judge that judge:

      * `n_archetypes` archetypes, drawn independently per item and per context
        from a Dirichlet.  Drawing them per context is the plan's
        "context-specific shift": the archetypes move, the mixing does not.
      * every remaining judge is an exact convex combination of the archetypes,
        with Dirichlet mixing weights.  These are the "convex interior judges".
      * `n_duplicates` of the interior judges are copied exactly onto another
        interior slot, which plants the cyclic individual certificates: a copy
        and its original each reconstruct the other at zero error, so each is
        individually removable and the pair is not jointly removable.

    Noise is a convex step toward a fresh distribution,
    ``p -> (1 - noise) p + noise u``, which keeps every row on the simplex with
    no clipping.  `noise_correlation` mixes a per-item shared draw into `u`
    against a judge-private one.  The shared part is an affine map applied to
    every judge alike, so it perturbs the panel without breaking any convex
    combination; only the private part actually damages reconstruction.  That
    is the point of the axis, and at ``noise_correlation = 1`` the planted
    backbone survives exactly however large `noise` is.

    Why the planted backbone is the *unique* minimum at zero tolerance, which
    is what makes recovery a real test rather than a plausibility check:
    stacked over items and classes, each archetype is a generic point in
    ``R^(3 * n_items)``, and ``3 * n_items`` far exceeds `n_archetypes`, so with
    probability one no archetype lies in the convex hull of the others.  A
    point outside the hull of the rest can only be reconstructed by itself, so
    every archetype must be retained; and the archetypes reconstruct every
    other judge by construction, so retaining them suffices.

    Returns the panel and its ground truth, never the ground truth alone, so a
    caller cannot accidentally select on the answer.
    """
    if n_archetypes < 2:
        raise ValueError("need at least 2 archetypes")
    if n_judges < n_archetypes:
        raise ValueError(
            f"n_judges {n_judges} cannot be below n_archetypes {n_archetypes}"
        )
    if not 0.0 <= noise <= 1.0:
        raise ValueError(f"noise {noise} must lie in [0, 1]")
    if not 0.0 <= noise_correlation <= 1.0:
        raise ValueError(f"noise_correlation {noise_correlation} must lie in [0, 1]")

    n_interior = n_judges - n_archetypes
    if n_duplicates > max(0, n_interior - 1):
        raise ValueError(
            f"n_duplicates {n_duplicates} needs {n_duplicates + 1} interior "
            f"judges but only {n_interior} exist"
        )

    rng = np.random.default_rng(seed)

    # Which slots the archetypes occupy.  Permuted so that "recover the
    # backbone" cannot be passed by returning the first few indices.
    order = rng.permutation(n_judges)
    backbone = sorted(int(i) for i in order[:n_archetypes])
    interior = sorted(int(i) for i in order[n_archetypes:])

    # Mixing weights: row j says how judge j is built from the archetypes.
    mixing = np.zeros((n_judges, n_archetypes), dtype=float)
    for slot, judge in enumerate(backbone):
        mixing[judge, slot] = 1.0
    if interior:
        mixing[interior, :] = rng.dirichlet(np.ones(n_archetypes), size=len(interior))

    # Exact copies, planting the dependency cycles.
    duplicate_of = {}
    for offset in range(n_duplicates):
        source, target = interior[offset], interior[offset + 1]
        mixing[target, :] = mixing[source, :]
        duplicate_of[target] = source

    context_names = ["clean"] + [f"shift_{c}" for c in range(1, n_contexts)]

    splits = {}
    for split in ("FIT", "CERT", "TEST"):
        blocks = []
        for _ in range(n_contexts):
            archetypes = _dirichlet_rows(rng, (n_items, n_archetypes), concentration)
            # (n_judges, n_arch) x (n_items, n_arch, 3) -> (n_items, n_judges, 3)
            block = np.einsum("ja,iax->ijx", mixing, archetypes)

            if noise > 0.0:
                shared = _dirichlet_rows(rng, (n_items, 1), 1.0)
                private = _dirichlet_rows(rng, (n_items, n_judges), 1.0)
                u = noise_correlation * shared + (1.0 - noise_correlation) * private
                block = (1.0 - noise) * block + noise * u

            blocks.append(block)
        splits[split] = JudgeResponses(blocks, context_names)

    names = [
        f"A{backbone.index(i)}" if i in set(backbone) else f"int{i:02d}"
        for i in range(n_judges)
    ]
    return ArchetypePanel(splits, names, backbone, mixing, duplicate_of)
