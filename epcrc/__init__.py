# epcrc/__init__.py
"""
EPCRC: Ecosystem Pruning via Convex Routing Coverage

A framework for pruning redundant models from AI ecosystems using 
convex routing coverage certificates.

Based on: "Ecosystem Pruning via Convex Routing Coverage" (2026)
"""

from epcrc.core import ModelUnit, Intervention, Scalarizer
from epcrc.geometry import DISCOSolver
from epcrc.ecosystem import Ecosystem
from epcrc.coverage import CoverageFunctional
from epcrc.pruning import BackwardElimination, ForwardSelection

__version__ = "0.1.0"
__all__ = [
    "ModelUnit",
    "Intervention",
    "Scalarizer",
    "DISCOSolver",
    "Ecosystem",
    "CoverageFunctional",
    "BackwardElimination",
    "ForwardSelection",
]
