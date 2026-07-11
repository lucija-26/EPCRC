# Graph Report - EPCRC  (2026-07-11)

## Corpus Check
- 41 files · ~337,302 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 296 nodes · 689 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `318e444f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_BackwardEliminationPruner|BackwardEliminationPruner]]
- [[_COMMUNITY_build_or_load_bundle|build_or_load_bundle]]
- [[_COMMUNITY_CoverageFunctional|CoverageFunctional]]
- [[_COMMUNITY_test_sparse_and_risk.py|test_sparse_and_risk.py]]
- [[_COMMUNITY_backward_elimination_hf.py|backward_elimination_hf.py]]
- [[_COMMUNITY_Research notes — MILP, the beta constraint, and the road to the report|Research notes — MILP, the beta constraint, and the road to the report]]
- [[_COMMUNITY_milp_min_representative_set|milp_min_representative_set]]
- [[_COMMUNITY_test_task_error.py|test_task_error.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_coverage.py|coverage.py]]
- [[_COMMUNITY_experiment_0_warm_forward.py|experiment_0_warm_forward.py]]
- [[_COMMUNITY_Math companion — every formula in the project, written out completely|Math companion — every formula in the project, written out completely]]
- [[_COMMUNITY_run_sweep_all.sh|run_sweep_all.sh]]
- [[_COMMUNITY_CLAUDE|CLAUDE.md]]
- [[_COMMUNITY_README|README.md]]

## God Nodes (most connected - your core abstractions)
1. `CoverageFunctional` - 80 edges
2. `BackwardEliminationPruner` - 35 edges
3. `ForwardSelectionPruner` - 30 edges
4. `PriorityQueuePruner` - 22 edges
5. `build_or_load_bundle()` - 19 edges
6. `BackwardKSwapPruner` - 17 edges
7. `SubstitutionCertificate` - 16 edges
8. `milp_min_representative_set()` - 15 edges
9. `PruningStep` - 15 edges
10. `step_metrics()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `ExactSearcher` --uses--> `CoverageFunctional`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/coverage.py
- `main()` --indirect_call--> `ForwardSelectionPruner`  [INFERRED]
  experiments/experiment_0_gamma_sweep_all.py → epcrc/pruning.py
- `main()` --indirect_call--> `ForwardSelectionPruner`  [INFERRED]
  experiments/experiment_0_gamma_sweep.py → epcrc/pruning.py
- `ExactSearcher` --uses--> `ForwardSelectionPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py
- `main()` --indirect_call--> `BackwardEliminationPruner`  [INFERRED]
  experiments/experiment_0_gamma_sweep_all.py → epcrc/pruning.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "BackwardEliminationPruner"
Cohesion: 0.09
Nodes (32): BackwardEliminationPruner, ForwardSelectionPruner, Section 4.1 backward elimination.      Start with S = J and remove one model at, Section 4.2 forward selection.      Start with S = empty and greedily add the mo, main(), plot_instance(), ndarray, Re-run backward and k-swap on separator instances found by experiments/experimen (+24 more)

### Community 1 - "build_or_load_bundle"
Cohesion: 0.11
Nodes (28): DataFrame, Detailed per-step metrics for coverage-based pruning.  Given a CoverageFunctiona, Compute a rich per-step metrics dict for a kept set S.      Returns a JSON-seria, step_metrics(), build_or_load_bundle(), build_xy(), _load_raw_csv(), prepare_utd19() (+20 more)

### Community 2 - "CoverageFunctional"
Cohesion: 0.12
Nodes (22): CoverageFunctional, ndarray, Sum of U(i|S) over all models i., Certificate for substituting one model using kept set S., Compute per-model substitution errors U(i|S) and coverage E(S)., SubstitutionCertificate, BackwardKSwapPruner, ForwardKSwapPruner (+14 more)

### Community 3 - "test_sparse_and_risk.py"
Cohesion: 0.11
Nodes (21): PruningResult, PruningStep, Greedy backward sweep: remove any model whose removal keeps E(S) <= gamma., Remove-k-add-(k-1) local search to shrink |S| past the phase-2 local minimum., coverage_ucb(), Risk-controlled pruning (paper Open Problem 1 / section 4.3, Experiment 4).  Poi, Per-model UCB on U(i|S), union-bounded over the N models., E_UCB(S) = max_i UCB(i|S). (+13 more)

### Community 4 - "backward_elimination_hf.py"
Cohesion: 0.18
Nodes (18): build_query_grid(), HFTextModel, mask_text(), ndarray, query_all_models(), EPCRC backward elimination on real Hugging Face text models.  Provides reusable, Build response matrix Y, shape (n_queries, n_models)., Make duplicate names explicit: x, x#2, x#3, ... (+10 more)

### Community 5 - "Research notes — MILP, the beta constraint, and the road to the report"
Cohesion: 0.09
Nodes (21): 1. Where the project stands (results already in the repo), 2. To run on the server when it's back up, 3.1 What a MILP is, 3.2 How ecosystem pruning becomes a MILP (`epcrc/milp.py`), 3.3 The one thing the MILP canNOT encode — and why that's still useful, 3.4 The oracle–protocol gap is itself a finding, 3.5 Practical solver notes, 3. MILP, explained from scratch (+13 more)

### Community 6 - "milp_min_representative_set"
Cohesion: 0.14
Nodes (18): milp_min_representative_set(), MilpResult, ndarray, MILP formulation of the minimal representative set problem (Open Problem 4).  So, Exact oracle-routing minimum representative set via MILP (HiGHS).      Variable, main(), MILP oracle-routing optimum on UTD19 subsets (Open Problem 4 / whiteboard OP4)., exact_opt() (+10 more)

### Community 7 - "test_task_error.py"
Cohesion: 0.19
Nodes (19): beta_coverage(), joint_feasible(), ndarray, quality_eligible_set(), Task-error preservation (the "beta" constraint).  The gamma constraint bounds *b, Models whose OWN task error on the shared eval sample is <= beta.      The quali, Feasibility under BOTH constraints: E(S) <= gamma and B(S) <= beta.      Drop-in, Per-model (err_orig, err_sub, delta) on the eval sample.      Models inside S su (+11 more)

### Community 8 - "__init__.py"
Cohesion: 0.21
Nodes (10): ABC, Intervention, ModelUnit, Any, Scalarizer, Ecosystem, Any, ndarray (+2 more)

### Community 9 - "coverage.py"
Cohesion: 0.16
Nodes (12): DISCOSolver, ndarray, Solve min ||target - peers @ w||^2 s.t. w >= 0, sum(w) = 1 via SLSQP.      Only, Simplex-constrained projection utilities.      For target vector y and peer matr, _slsqp_simplex(), ExactSearcher, main(), ndarray (+4 more)

### Community 10 - "experiment_0_warm_forward.py"
Cohesion: 0.22
Nodes (11): Warm-started forward pruning seeded with an initial kept set.      Starts from a, WarmStartForwardWorstCoveredPruner, aggregate(), farthest_pair_mean_abs(), main(), print_size_table(), ndarray, Task 3: UTD19 warm-forward gamma sweep with random subset runs.  Compares three (+3 more)

### Community 11 - "Math companion — every formula in the project, written out completely"
Cohesion: 0.25
Nodes (7): 1. The objects everything is built from, 2. The algorithms as formulas, 3. The MILP, written out completely, 4. Beta (task-error preservation), with full derivations, 5. Risk-controlled pruning (UCB), 6. Carathéodory: the geometry that predicts your numbers, Math companion — every formula in the project, written out completely

### Community 12 - "run_sweep_all.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_sweep_all.sh script

## Knowledge Gaps
- **30 isolated node(s):** `run_sweep_all.sh script`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `graphify` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CoverageFunctional` connect `CoverageFunctional` to `BackwardEliminationPruner`, `build_or_load_bundle`, `test_sparse_and_risk.py`, `backward_elimination_hf.py`, `milp_min_representative_set`, `test_task_error.py`, `__init__.py`, `coverage.py`, `experiment_0_warm_forward.py`?**
  _High betweenness centrality (0.361) - this node is a cross-community bridge._
- **Why does `BackwardEliminationPruner` connect `BackwardEliminationPruner` to `build_or_load_bundle`, `CoverageFunctional`, `test_sparse_and_risk.py`, `backward_elimination_hf.py`, `milp_min_representative_set`, `__init__.py`, `coverage.py`, `experiment_0_warm_forward.py`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `ForwardSelectionPruner` connect `BackwardEliminationPruner` to `build_or_load_bundle`, `CoverageFunctional`, `test_sparse_and_risk.py`, `backward_elimination_hf.py`, `__init__.py`, `coverage.py`, `experiment_0_warm_forward.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `CoverageFunctional` (e.g. with `HFTextModel` and `DISCOSolver`) actually correct?**
  _`CoverageFunctional` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `BackwardEliminationPruner` (e.g. with `HFTextModel` and `CoverageFunctional`) actually correct?**
  _`BackwardEliminationPruner` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `ForwardSelectionPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`ForwardSelectionPruner` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `PriorityQueuePruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`PriorityQueuePruner` has 5 INFERRED edges - model-reasoned connections that need verification._