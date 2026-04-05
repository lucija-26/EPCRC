"""EPCRC: Ecosystem Pruning via Convex Routing Coverage.

Minimal implementation for Section 4.1 backward elimination.
"""

from .core import Intervention, ModelUnit, Scalarizer
from .coverage import CoverageFunctional, SubstitutionCertificate
from .ecosystem import Ecosystem
from .pruning import BackwardEliminationPruner, ForwardSelectionPruner, PruningResult, PruningStep

__all__ = [
    "CoverageFunctional",
    "SubstitutionCertificate",
    "Intervention",
    "ModelUnit",
    "Scalarizer",
    "Ecosystem",
    "BackwardEliminationPruner",
    "ForwardSelectionPruner",
    "PruningResult",
    "PruningStep",
]
