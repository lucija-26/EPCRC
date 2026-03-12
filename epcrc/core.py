from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Intervention(ABC):
    @abstractmethod
    def apply(self, x: Any, theta: float, seed: Optional[int] = None) -> Any:
        raise NotImplementedError


class Scalarizer(ABC):
    @abstractmethod
    def __call__(self, raw_output: Any) -> float:
        raise NotImplementedError


class ModelUnit(ABC):
    def __init__(self, name: str, scalarizer: Optional[Scalarizer] = None):
        self.name = name
        self.scalarizer = scalarizer

    @abstractmethod
    def _forward(self, input_data: Any) -> Any:
        raise NotImplementedError

    def query(self, x: Any, theta: float, intervention: Intervention, seed: Optional[int] = None) -> float:
        try:
            perturbed = intervention.apply(x, theta, seed)
        except TypeError:
            perturbed = intervention.apply(x, theta)

        raw = self._forward(perturbed)
        if self.scalarizer is None:
            return float(raw)
        return float(self.scalarizer(raw))
