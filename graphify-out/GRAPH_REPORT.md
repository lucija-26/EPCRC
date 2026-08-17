# Graph Report - EPCRC  (2026-08-17)

## Corpus Check
- 31 files · ~30,986 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 275 nodes · 616 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5ce01510`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_BackwardEliminationPruner|BackwardEliminationPruner]]
- [[_COMMUNITY_CoverageFunctional|CoverageFunctional]]
- [[_COMMUNITY_test_sparse_and_risk.py|test_sparse_and_risk.py]]
- [[_COMMUNITY_Research notes — MILP, the beta constraint, and the road to the report|Research notes — MILP, the beta constraint, and the road to the report]]
- [[_COMMUNITY_milp_min_representative_set|milp_min_representative_set]]
- [[_COMMUNITY_test_task_error.py|test_task_error.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_coverage.py|coverage.py]]
- [[_COMMUNITY_PriorityQueuePruner|PriorityQueuePruner]]
- [[_COMMUNITY_Math companion — every formula in the project, written out completely|Math companion — every formula in the project, written out completely]]
- [[_COMMUNITY_CLAUDE|CLAUDE.md]]
- [[_COMMUNITY_README|README.md]]
- [[_COMMUNITY_experiment_0_warm_forward.py|experiment_0_warm_forward.py]]

## God Nodes (most connected - your core abstractions)
1. `CoverageFunctional` - 69 edges
2. `BackwardEliminationPruner` - 29 edges
3. `ForwardSelectionPruner` - 26 edges
4. `PriorityQueuePruner` - 21 edges
5. `BackwardKSwapPruner` - 18 edges
6. `SubstitutionCertificate` - 16 edges
7. `milp_min_representative_set()` - 16 edges
8. `PruningStep` - 16 edges
9. `PruningResult` - 14 edges
10. `ExactSearcher` - 12 edges

## Surprising Connections (you probably didn't know these)
- `ExactSearcher` --uses--> `CoverageFunctional`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/coverage.py
- `main()` --calls--> `CoverageFunctional`  [EXTRACTED]
  experiments/experiment_subset_robustness.py → epcrc/coverage.py
- `ExactSearcher` --uses--> `ForwardSelectionPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py
- `ExactSearcher` --uses--> `BackwardEliminationPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py
- `ExactSearcher` --uses--> `PriorityQueuePruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "BackwardEliminationPruner"
Cohesion: 0.09
Nodes (32): BackwardEliminationPruner, ForwardSelectionPruner, Section 4.1 backward elimination.      Start with S = J and remove one model at, Section 4.2 forward selection.      Start with S = empty and greedily add the mo, main(), plot_instance(), ndarray, Re-run backward and k-swap on separator instances found by experiments/experimen (+24 more)

### Community 2 - "CoverageFunctional"
Cohesion: 0.09
Nodes (33): _cert_check(), _cert_init(), CertificateResult, certify_lower_bound(), is_feasible_set(), milp_min_representative_set(), MilpResult, min_substitution_error() (+25 more)

### Community 3 - "test_sparse_and_risk.py"
Cohesion: 0.15
Nodes (17): coverage_ucb(), Risk-controlled pruning (paper Open Problem 1 / section 4.3, Experiment 4).  Poi, Per-model UCB on U(i|S), union-bounded over the N models., E_UCB(S) = max_i UCB(i|S)., Backward elimination that accepts a removal only when E_UCB <= gamma.      Stric, RiskControlledBackwardPruner, uniqueness_ucb(), _noisy_instance() (+9 more)

### Community 5 - "Research notes — MILP, the beta constraint, and the road to the report"
Cohesion: 0.09
Nodes (21): 1. Where the project stands (results already in the repo), 2. To run on the server when it's back up, 3.1 What a MILP is, 3.2 How ecosystem pruning becomes a MILP (`epcrc/milp.py`), 3.3 The one thing the MILP canNOT encode — and why that's still useful, 3.4 The oracle–protocol gap is itself a finding, 3.5 Practical solver notes, 3. MILP, explained from scratch (+13 more)

### Community 6 - "milp_min_representative_set"
Cohesion: 0.39
Nodes (7): audit_disco(), audit_monotonicity(), check(), main(), ndarray, Correctness audit of every pruner + the DISCO solver, on real UTD19 data.  Check, KKT check of the simplex-constrained least squares fit.

### Community 7 - "test_task_error.py"
Cohesion: 0.14
Nodes (25): CoverageFunctional, ndarray, Compute per-model substitution errors U(i|S) and coverage E(S)., Detailed per-step metrics for coverage-based pruning.  Given a CoverageFunctiona, Compute a rich per-step metrics dict for a kept set S.      Returns a JSON-seria, step_metrics(), beta_coverage(), joint_feasible() (+17 more)

### Community 8 - "__init__.py"
Cohesion: 0.21
Nodes (10): ABC, Intervention, ModelUnit, Any, Scalarizer, Ecosystem, Any, ndarray (+2 more)

### Community 9 - "coverage.py"
Cohesion: 0.11
Nodes (18): Sum of U(i|S) over all models i., Certificate for substituting one model using kept set S., SubstitutionCertificate, DISCOSolver, ndarray, Mask divide/overflow/invalid flags raised by the BLAS matmul kernel.      numpy, Solve min ||target - peers @ w||^2 s.t. w >= 0, sum(w) = 1 via SLSQP.      Only, Simplex-constrained projection utilities.      For target vector y and peer matr (+10 more)

### Community 10 - "PriorityQueuePruner"
Cohesion: 0.11
Nodes (18): BackwardKSwapPruner, ForwardKSwapPruner, PriorityQueuePruner, PruningResult, PruningStep, Lazy greedy forward + backward cleanup, with optional k-swap escape.      Phase, Greedy backward sweep: remove any model whose removal keeps E(S) <= gamma., Remove-k-add-(k-1) local search to shrink |S| past the phase-2 local minimum. (+10 more)

### Community 11 - "Math companion — every formula in the project, written out completely"
Cohesion: 0.25
Nodes (7): 1. The objects everything is built from, 2. The algorithms as formulas, 3. The MILP, written out completely, 4. Beta (task-error preservation), with full derivations, 5. Risk-controlled pruning (UCB), 6. Carathéodory: the geometry that predicts your numbers, Math companion — every formula in the project, written out completely

### Community 15 - "experiment_0_warm_forward.py"
Cohesion: 0.13
Nodes (22): DataFrame, Warm-started forward pruning seeded with an initial kept set.      Starts from a, WarmStartForwardWorstCoveredPruner, build_or_load_bundle(), build_xy(), _load_raw_csv(), prepare_utd19(), ndarray (+14 more)

## Knowledge Gaps
- **26 isolated node(s):** `graphify`, `EPCRC`, `1. The objects everything is built from`, `2. The algorithms as formulas`, `3. The MILP, written out completely` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CoverageFunctional` connect `test_task_error.py` to `BackwardEliminationPruner`, `CoverageFunctional`, `test_sparse_and_risk.py`, `milp_min_representative_set`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `experiment_0_warm_forward.py`?**
  _High betweenness centrality (0.332) - this node is a cross-community bridge._
- **Why does `BackwardEliminationPruner` connect `BackwardEliminationPruner` to `CoverageFunctional`, `test_sparse_and_risk.py`, `milp_min_representative_set`, `test_task_error.py`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `experiment_0_warm_forward.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `ForwardSelectionPruner` connect `BackwardEliminationPruner` to `CoverageFunctional`, `milp_min_representative_set`, `test_task_error.py`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `experiment_0_warm_forward.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `CoverageFunctional` (e.g. with `DISCOSolver` and `BackwardEliminationPruner`) actually correct?**
  _`CoverageFunctional` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `BackwardEliminationPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`BackwardEliminationPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ForwardSelectionPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`ForwardSelectionPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `PriorityQueuePruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`PriorityQueuePruner` has 3 INFERRED edges - model-reasoned connections that need verification._