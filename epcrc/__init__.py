"""EPCRC: Ecosystem Pruning via Convex Routing Coverage.

Minimal implementation for Section 4.1 backward elimination.
"""

from .core import Intervention, ModelUnit, Scalarizer
from .coverage import CoverageFunctional, SubstitutionCertificate
from .ecosystem import Ecosystem
from .milp import MilpResult, milp_min_representative_set
from .task_error import (
    beta_coverage,
    joint_feasible,
    quality_eligible_set,
    substitution_task_errors,
)
from .pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
    PruningResult,
    PruningStep,
    WarmStartForwardWorstCoveredPruner,
)

__all__ = [
    "CoverageFunctional",
    "SubstitutionCertificate",
    "Intervention",
    "ModelUnit",
    "Scalarizer",
    "Ecosystem",
    "BackwardEliminationPruner",
    "BackwardKSwapPruner",
    "ForwardKSwapPruner",
    "ForwardSelectionPruner",
    "WarmStartForwardWorstCoveredPruner",
    "PriorityQueuePruner",
    "PruningResult",
    "PruningStep",
    "MilpResult",
    "milp_min_representative_set",
    "beta_coverage",
    "joint_feasible",
    "quality_eligible_set",
    "substitution_task_errors",
]
