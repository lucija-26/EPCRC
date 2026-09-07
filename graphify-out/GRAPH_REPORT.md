# Graph Report - EPCRC  (2026-09-07)

## Corpus Check
- 42 files · ~62,706 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 504 nodes · 1143 edges · 27 communities (22 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `51a43e6c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_BackwardEliminationPruner|BackwardEliminationPruner]]
- [[_COMMUNITY_figs.py|figs.py]]
- [[_COMMUNITY_CoverageFunctional|CoverageFunctional]]
- [[_COMMUNITY_test_sparse_and_risk.py|test_sparse_and_risk.py]]
- [[_COMMUNITY_2. Algorithms|2. Algorithms]]
- [[_COMMUNITY_Research notes — MILP, the beta constraint, and the road to the report|Research notes — MILP, the beta constraint, and the road to the report]]
- [[_COMMUNITY_milp_min_representative_set|milp_min_representative_set]]
- [[_COMMUNITY_test_task_error.py|test_task_error.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_coverage.py|coverage.py]]
- [[_COMMUNITY_PriorityQueuePruner|PriorityQueuePruner]]
- [[_COMMUNITY_Math companion — every formula in the project, written out completely|Math companion — every formula in the project, written out completely]]
- [[_COMMUNITY_CoverageFunctional|CoverageFunctional]]
- [[_COMMUNITY_CLAUDE|CLAUDE.md]]
- [[_COMMUNITY_README|README.md]]
- [[_COMMUNITY_experiment_0_warm_forward.py|experiment_0_warm_forward.py]]
- [[_COMMUNITY_Callout|Callout]]
- [[_COMMUNITY_Doc|Doc]]
- [[_COMMUNITY_build_or_load_bundle|build_or_load_bundle]]
- [[_COMMUNITY_Para|Para]]
- [[_COMMUNITY_doclib.py|doclib.py]]
- [[_COMMUNITY_experiment_forward_beats_backward.py|experiment_forward_beats_backward.py]]
- [[_COMMUNITY_measure|measure]]
- [[_COMMUNITY_Block|Block]]
- [[_COMMUNITY_build.py|build.py]]
- [[_COMMUNITY_Space|Space]]
- [[_COMMUNITY_Bullet|Bullet]]

## God Nodes (most connected - your core abstractions)
1. `CoverageFunctional` - 72 edges
2. `BackwardEliminationPruner` - 31 edges
3. `ForwardSelectionPruner` - 26 edges
4. `panels()` - 21 edges
5. `JudgeResponses` - 21 edges
6. `PriorityQueuePruner` - 21 edges
7. `SubstitutionCertificate` - 20 edges
8. `Doc` - 18 edges
9. `simplex_ax()` - 18 edges
10. `JudgeCoverageFunctional` - 18 edges

## Surprising Connections (you probably didn't know these)
- `ExactSearcher` --uses--> `CoverageFunctional`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/coverage.py
- `ExactSearcher` --uses--> `ForwardSelectionPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py
- `ExactSearcher` --uses--> `BackwardEliminationPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py
- `ExactSearcher` --uses--> `PriorityQueuePruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py
- `ExactSearcher` --uses--> `BackwardKSwapPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py

## Import Cycles
- None detected.

## Communities (27 total, 5 thin omitted)

### Community 0 - "BackwardEliminationPruner"
Cohesion: 0.12
Nodes (23): BackwardEliminationPruner, ForwardSelectionPruner, Section 4.1 backward elimination.      Start with S = J and remove one model at, Section 4.2 forward selection.      Start with S = empty and greedily add the mo, main(), plot_instance(), ndarray, Re-run backward and k-swap on separator instances found by experiments/experimen (+15 more)

### Community 1 - "figs.py"
Cohesion: 0.12
Nodes (55): cloud_panel(), main(), One figure: how EPCRC and the LLM judge panel are the same problem.  Writes docs, Draw a row of coordinate cells; group cells into blocks of `group`., vector_strip(), bare_ax(), _ccw(), _circle_judges() (+47 more)

### Community 2 - "CoverageFunctional"
Cohesion: 0.07
Nodes (41): _cert_check(), _cert_init(), CertificateResult, certify_lower_bound(), is_feasible_set(), milp_min_representative_set(), MilpResult, min_substitution_error() (+33 more)

### Community 3 - "test_sparse_and_risk.py"
Cohesion: 0.20
Nodes (13): Mask divide/overflow/invalid flags raised by the BLAS matmul kernel.      numpy, suppress_spurious_blas_flags(), coverage_ucb(), Risk-controlled pruning (paper Open Problem 1 / section 4.3, Experiment 4).  Poi, Per-model UCB on U(i|S), union-bounded over the N models., E_UCB(S) = max_i UCB(i|S)., uniqueness_ucb(), errstate (+5 more)

### Community 4 - "2. Algorithms"
Cohesion: 0.11
Nodes (17): 1. Problem statement, 2.1 Forward selection (`forward`), 2.2 Forward selection with trimming (`forward_trim`), 2.3 Backward elimination (`backward`), 2.4 $k$-swap (`backward_kswap2`, `backward_kswap3`), 2.5 Priority-queue $k$-swap (`pq_kswap`), 2.6 Complexity, 2. Algorithms (+9 more)

### Community 5 - "Research notes — MILP, the beta constraint, and the road to the report"
Cohesion: 0.09
Nodes (21): 1. Where the project stands (results already in the repo), 2. To run on the server when it's back up, 3.1 What a MILP is, 3.2 How ecosystem pruning becomes a MILP (`epcrc/milp.py`), 3.3 The one thing the MILP canNOT encode — and why that's still useful, 3.4 The oracle–protocol gap is itself a finding, 3.5 Practical solver notes, 3. MILP, explained from scratch (+13 more)

### Community 6 - "milp_min_representative_set"
Cohesion: 0.20
Nodes (13): BackwardKSwapPruner, PriorityQueuePruner, Lazy greedy forward + backward cleanup, with optional k-swap escape.      Phase, Backward elimination seed + the k-swap reduction escape.      Backward eliminati, audit_disco(), audit_monotonicity(), check(), main() (+5 more)

### Community 7 - "test_task_error.py"
Cohesion: 0.19
Nodes (19): beta_coverage(), joint_feasible(), ndarray, quality_eligible_set(), Task-error preservation (the "beta" constraint).  The gamma constraint bounds *b, Models whose OWN task error on the shared eval sample is <= beta.      The quali, Feasibility under BOTH constraints: E(S) <= gamma and B(S) <= beta.      Drop-in, Per-model (err_orig, err_sub, delta) on the eval sample.      Models inside S su (+11 more)

### Community 8 - "__init__.py"
Cohesion: 0.22
Nodes (9): ABC, Intervention, ModelUnit, Any, Scalarizer, Ecosystem, Any, ndarray (+1 more)

### Community 9 - "coverage.py"
Cohesion: 0.16
Nodes (12): DISCOSolver, ndarray, Solve min ||target - peers @ w||^2 s.t. w >= 0, sum(w) = 1 via SLSQP.      Only, Simplex-constrained projection utilities.      For target vector y and peer matr, _slsqp_simplex(), ExactSearcher, main(), ndarray (+4 more)

### Community 10 - "PriorityQueuePruner"
Cohesion: 0.22
Nodes (7): PruningResult, PruningStep, Greedy backward sweep: remove any model whose removal keeps E(S) <= gamma., Remove-k-add-(k-1) local search to shrink |S| past the phase-2 local minimum., Trim redundant models, then package the result., Backward elimination that accepts a removal only when E_UCB <= gamma.      Stric, RiskControlledBackwardPruner

### Community 11 - "Math companion — every formula in the project, written out completely"
Cohesion: 0.25
Nodes (7): 1. The objects everything is built from, 2. The algorithms as formulas, 3. The MILP, written out completely, 4. Beta (task-error preservation), with full derivations, 5. Risk-controlled pruning (UCB), 6. Carathéodory: the geometry that predicts your numbers, Math companion — every formula in the project, written out completely

### Community 12 - "CoverageFunctional"
Cohesion: 0.14
Nodes (13): CoverageFunctional, ndarray, Sum of U(i|S) over all models i., Certificate for substituting one model using kept set S., Compute per-model substitution errors U(i|S) and coverage E(S)., SubstitutionCertificate, Detailed per-step metrics for coverage-based pruning.  Given a CoverageFunctiona, Compute a rich per-step metrics dict for a kept set S.      Returns a JSON-seria (+5 more)

### Community 15 - "experiment_0_warm_forward.py"
Cohesion: 0.12
Nodes (22): DataFrame, Warm-started forward pruning seeded with an initial kept set.      Starts from a, WarmStartForwardWorstCoveredPruner, build_or_load_bundle(), build_xy(), _load_raw_csv(), prepare_utd19(), ndarray (+14 more)

### Community 16 - "Callout"
Cohesion: 0.16
Nodes (3): Callout, FigBlock, PageBreak

### Community 17 - "Doc"
Cohesion: 0.22
Nodes (3): Doc, Heading, Flow blocks onto pages. Returns {heading label: page number}.

### Community 18 - "build_or_load_bundle"
Cohesion: 0.06
Nodes (64): EPCRC: Ecosystem Pruning via Convex Routing Coverage.  Minimal implementation fo, context_errors(), JudgeCoverageFunctional, JudgeResponses, ndarray, Three-class judge outputs: total-variation loss and worst-context fitting.  An L, Total variation between three-class rows, along the last axis., Mean TV error of the reconstruction of `target_idx`, one value per context. (+56 more)

### Community 19 - "Para"
Cohesion: 0.24
Nodes (3): Para, _PreWrapped, Return (head, tail) blocks so that head fits in ``avail`` inches.

### Community 21 - "doclib.py"
Cohesion: 0.24
Nodes (7): _emit_span(), _emit_words(), Minimal flowing-document engine on top of matplotlib's PDF backend.  Produces a, Split into (text, weight, style) tokens.      ``**bold**`` and ``__italic__`` ar, Emit words from ``chunk``, keeping ``$...$`` spans as single tokens., tokenize(), Contact-sheet renderer: draws every figure in figs.py to _png/ for review.

### Community 22 - "experiment_forward_beats_backward.py"
Cohesion: 0.29
Nodes (9): main(), mean_pairwise_dist(), pick_smallest(), plot_instance(), ndarray, Task 2: find a SEPARATING instance where forward < backward.  The headline findi, Smallest = fewest models N, then largest gap, then dim=2 (plottable)., pts: (N, d). Mean Euclidean distance between distinct points. (+1 more)

### Community 23 - "measure"
Cohesion: 0.25
Nodes (3): Eq, measure(), Width and height of a rendered string, in inches.

### Community 26 - "build.py"
Cohesion: 0.60
Nodes (4): build(), cover(), make_toc(), Build JUDGE_PANEL_GEOMETRY.pdf -- a visual math companion to the LLM judge-panel

## Knowledge Gaps
- **39 isolated node(s):** `graphify`, `EPCRC`, `Routing semantics`, `2.1 Forward selection (`forward`)`, `2.2 Forward selection with trimming (`forward_trim`)` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CoverageFunctional` connect `CoverageFunctional` to `BackwardEliminationPruner`, `CoverageFunctional`, `test_sparse_and_risk.py`, `milp_min_representative_set`, `test_task_error.py`, `coverage.py`, `PriorityQueuePruner`, `experiment_0_warm_forward.py`, `build_or_load_bundle`, `experiment_forward_beats_backward.py`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `BackwardEliminationPruner` connect `BackwardEliminationPruner` to `CoverageFunctional`, `test_sparse_and_risk.py`, `milp_min_representative_set`, `coverage.py`, `PriorityQueuePruner`, `CoverageFunctional`, `experiment_0_warm_forward.py`, `build_or_load_bundle`, `experiment_forward_beats_backward.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `JudgeCoverageFunctional` connect `build_or_load_bundle` to `CoverageFunctional`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `CoverageFunctional` (e.g. with `DISCOSolver` and `JudgeCoverageFunctional`) actually correct?**
  _`CoverageFunctional` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `BackwardEliminationPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`BackwardEliminationPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ForwardSelectionPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`ForwardSelectionPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `JudgeResponses` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`JudgeResponses` has 2 INFERRED edges - model-reasoned connections that need verification._