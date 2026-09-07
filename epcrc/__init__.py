"""EPCRC: Ecosystem Pruning via Convex Routing Coverage.

Minimal implementation for Section 4.1 backward elimination.
"""

from .core import Intervention, ModelUnit, Scalarizer
from .coverage import CoverageFunctional, SubstitutionCertificate
from .ecosystem import Ecosystem
from .judge import (
    JudgeCoverageFunctional,
    JudgeResponses,
    context_errors,
    solve_minimax_weights,
    total_variation,
    worst_context_error,
)
from .milp import MilpResult, milp_min_representative_set
from .prompts import (
    REGISTERED_CONTEXTS,
    Context,
    canonical_gold_label,
    canonicalize,
    protocol_hashes,
    render,
)
from .rewardbench import (
    ALL_SEEDS,
    PRIMARY_SEED,
    JudgedPair,
    build_pairs,
    grouped_split,
    load_reward_bench_2,
    read_pairs,
    split_manifest,
    stratified_subset,
    write_pairs,
)
from .scoring import (
    LabelScorer,
    accuracy_report,
    argmax_labels,
    normalize_label_scores,
    score_pairs,
)
from .synthetic_judges import curved_arc, duplicated_extremes, sagitta, to_simplex
from .risk import RiskControlledBackwardPruner, coverage_ucb, uniqueness_ucb
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
    "JudgeCoverageFunctional",
    "JudgeResponses",
    "context_errors",
    "solve_minimax_weights",
    "total_variation",
    "worst_context_error",
    "curved_arc",
    "duplicated_extremes",
    "sagitta",
    "to_simplex",
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
    "RiskControlledBackwardPruner",
    "coverage_ucb",
    "uniqueness_ucb",
    "REGISTERED_CONTEXTS",
    "Context",
    "canonical_gold_label",
    "canonicalize",
    "protocol_hashes",
    "render",
    "ALL_SEEDS",
    "PRIMARY_SEED",
    "JudgedPair",
    "build_pairs",
    "grouped_split",
    "load_reward_bench_2",
    "read_pairs",
    "split_manifest",
    "stratified_subset",
    "write_pairs",
    "LabelScorer",
    "accuracy_report",
    "argmax_labels",
    "normalize_label_scores",
    "score_pairs",
]
