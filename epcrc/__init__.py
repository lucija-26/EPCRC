"""EPCRC: Ecosystem Pruning via Convex Routing Coverage.

Certified compression of LLM judge panels by joint behavioural coverage.
"""

from .coverage import CoverageFunctional, SubstitutionCertificate
from .judge import (
    JudgeCoverageFunctional,
    JudgeResponses,
    context_errors,
    solve_minimax_weights,
    solve_minimax_weights_cuts,
    solve_minimax_weights_lp,
    total_variation,
    worst_context_error,
)
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
from .pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardKSwapPruner,
    ForwardSelectionPruner,
    PriorityQueuePruner,
    PruningResult,
    PruningStep,
)

__all__ = [
    "CoverageFunctional",
    "SubstitutionCertificate",
    "JudgeCoverageFunctional",
    "JudgeResponses",
    "context_errors",
    "solve_minimax_weights",
    "solve_minimax_weights_cuts",
    "solve_minimax_weights_lp",
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
    "PriorityQueuePruner",
    "PruningResult",
    "PruningStep",
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
