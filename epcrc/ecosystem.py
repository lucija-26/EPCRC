from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from .core import Intervention, ModelUnit


class Ecosystem:
    """Utility to query one target model and peer models on matched interventions."""

    def __init__(self, target: ModelUnit, peers: Sequence[ModelUnit]):
        self.target = target
        self.peers = list(peers)

    def batched_query(
        self,
        X: Iterable[Any],
        Thetas: Iterable[float],
        intervention: Intervention,
        seeds: Optional[Iterable[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = list(X)
        T_list = list(Thetas)
        if len(X_list) != len(T_list):
            raise ValueError("X and Thetas must have the same length")

        if seeds is None:
            seeds_list = [None] * len(X_list)
        else:
            seeds_list = list(seeds)
            if len(seeds_list) != len(X_list):
                raise ValueError("seeds length must match X")

        y_t = []
        Y_p = []

        for x, theta, seed in zip(X_list, T_list, seeds_list):
            try:
                perturbed = intervention.apply(x, float(theta), seed)
            except TypeError:
                perturbed = intervention.apply(x, float(theta))

            raw_t = self.target._forward(perturbed)
            val_t = self.target.scalarizer(raw_t) if self.target.scalarizer else raw_t
            y_t.append(float(val_t))

            peer_vals = []
            for p in self.peers:
                raw_p = p._forward(perturbed)
                val_p = p.scalarizer(raw_p) if p.scalarizer else raw_p
                peer_vals.append(float(val_p))
            Y_p.append(peer_vals)

        return np.asarray(y_t, dtype=float), np.asarray(Y_p, dtype=float)
