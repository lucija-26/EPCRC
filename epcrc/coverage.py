from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from geometry import DISCOSolver


@dataclass
class SubstitutionCertificate:
    """Certificate for substituting one model using kept set S."""

    model_idx: int
    model_name: str
    kept_set: Set[int]
    weights: np.ndarray
    uniqueness: float


class CoverageFunctional:
    """Compute per-model substitution errors U(i|S) and coverage E(S)."""

    def __init__(
        self,
        Y_fit: np.ndarray,
        Y_eval: np.ndarray,
        model_names: Optional[List[str]] = None,
        metric: str = "mean_abs",
    ):
        self.Y_fit = np.asarray(Y_fit, dtype=float)
        self.Y_eval = np.asarray(Y_eval, dtype=float)

        if self.Y_fit.ndim != 2 or self.Y_eval.ndim != 2:
            raise ValueError("Y_fit and Y_eval must be 2D arrays: (n_queries, n_models)")

        n_fit, n_models = self.Y_fit.shape
        n_eval, n_models_eval = self.Y_eval.shape

        if n_models != n_models_eval:
            raise ValueError(
                f"Model dimension mismatch: Y_fit has {n_models}, Y_eval has {n_models_eval}"
            )

        if n_fit == 0 or n_eval == 0:
            raise ValueError("Y_fit and Y_eval must have at least one row")

        self.N = n_models
        self.metric = metric
        self.model_names = model_names or [f"model_{i}" for i in range(self.N)]
        self._cache: Dict[frozenset, Dict[int, SubstitutionCertificate]] = {}

    def compute_certificate(self, target_idx: int, kept_set: Set[int]) -> SubstitutionCertificate:
        kept_set = set(kept_set)
        kept_list = sorted(kept_set)

        if target_idx in kept_set:
            w = np.zeros(len(kept_list), dtype=float)
            w[kept_list.index(target_idx)] = 1.0
            return SubstitutionCertificate(
                model_idx=target_idx,
                model_name=self.model_names[target_idx],
                kept_set=kept_set,
                weights=w,
                uniqueness=0.0,
            )

        if len(kept_list) == 0:
            return SubstitutionCertificate(
                model_idx=target_idx,
                model_name=self.model_names[target_idx],
                kept_set=kept_set,
                weights=np.array([], dtype=float),
                uniqueness=float("inf"),
            )

        y_fit = self.Y_fit[:, target_idx]
        Yp_fit = self.Y_fit[:, kept_list]

        _, w = DISCOSolver.solve_weights_and_distance(y_fit, Yp_fit)

        y_eval = self.Y_eval[:, target_idx]
        Yp_eval = self.Y_eval[:, kept_list]
        u = DISCOSolver.compute_uniqueness(y_eval, Yp_eval, w, metric=self.metric)

        return SubstitutionCertificate(
            model_idx=target_idx,
            model_name=self.model_names[target_idx],
            kept_set=kept_set,
            weights=w,
            uniqueness=u,
        )

    def compute_coverage(
        self,
        kept_set: Set[int],
        return_certificates: bool = False,
    ) -> Tuple[float, Optional[Dict[int, SubstitutionCertificate]]]:
        key = frozenset(kept_set)
        if key not in self._cache:
            certs: Dict[int, SubstitutionCertificate] = {}
            for i in range(self.N):
                certs[i] = self.compute_certificate(i, set(kept_set))
            self._cache[key] = certs

        certs = self._cache[key]
        coverage = float(max(c.uniqueness for c in certs.values()))

        if return_certificates:
            return coverage, certs
        return coverage, None

    def find_bottleneck(self, kept_set: Set[int]) -> SubstitutionCertificate:
        _, certs = self.compute_coverage(kept_set, return_certificates=True)
        assert certs is not None
        return max(certs.values(), key=lambda c: c.uniqueness)
