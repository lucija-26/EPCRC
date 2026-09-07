# Graph Report - EPCRC  (2026-08-29)

## Corpus Check
- 36 files · ~78,954 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 425 nodes · 960 edges · 29 communities (24 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c081b788`
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
- [[_COMMUNITY_test_pruning_synthetic.py|test_pruning_synthetic.py]]
- [[_COMMUNITY_doclib.py|doclib.py]]
- [[_COMMUNITY_experiment_forward_beats_backward.py|experiment_forward_beats_backward.py]]
- [[_COMMUNITY_measure|measure]]
- [[_COMMUNITY_Block|Block]]
- [[_COMMUNITY_SubstitutionCertificate|SubstitutionCertificate]]
- [[_COMMUNITY_build.py|build.py]]
- [[_COMMUNITY_Space|Space]]
- [[_COMMUNITY_Bullet|Bullet]]

## God Nodes (most connected - your core abstractions)
1. `CoverageFunctional` - 69 edges
2. `BackwardEliminationPruner` - 29 edges
3. `ForwardSelectionPruner` - 26 edges
4. `panels()` - 21 edges
5. `PriorityQueuePruner` - 21 edges
6. `Doc` - 18 edges
7. `BackwardKSwapPruner` - 18 edges
8. `simplex_ax()` - 17 edges
9. `to_xy()` - 16 edges
10. `SubstitutionCertificate` - 16 edges

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

## Communities (29 total, 5 thin omitted)

### Community 0 - "BackwardEliminationPruner"
Cohesion: 0.18
Nodes (13): BackwardEliminationPruner, ForwardSelectionPruner, Section 4.1 backward elimination.      Start with S = J and remove one model at, Section 4.2 forward selection.      Start with S = empty and greedily add the mo, main(), plot_instance(), ndarray, Re-run backward and k-swap on separator instances found by experiments/experimen (+5 more)

### Community 1 - "figs.py"
Cohesion: 0.14
Nodes (50): bare_ax(), _ccw(), _circle_judges(), clip_poly(), coverage(), dist_to_hull(), dot(), _eb_slack() (+42 more)

### Community 2 - "CoverageFunctional"
Cohesion: 0.08
Nodes (38): _cert_check(), _cert_init(), CertificateResult, certify_lower_bound(), is_feasible_set(), milp_min_representative_set(), MilpResult, min_substitution_error() (+30 more)

### Community 3 - "test_sparse_and_risk.py"
Cohesion: 0.27
Nodes (10): coverage_ucb(), Risk-controlled pruning (paper Open Problem 1 / section 4.3, Experiment 4).  Poi, Per-model UCB on U(i|S), union-bounded over the N models., E_UCB(S) = max_i UCB(i|S)., uniqueness_ucb(), _noisy_instance(), Sanity checks for sparse MILP certificates (OP5) and risk-controlled pruning (OP, UCB-gated backward keeps at least as many models as plain backward,     and its (+2 more)

### Community 4 - "2. Algorithms"
Cohesion: 0.11
Nodes (17): 1. Problem statement, 2.1 Forward selection (`forward`), 2.2 Forward selection with trimming (`forward_trim`), 2.3 Backward elimination (`backward`), 2.4 $k$-swap (`backward_kswap2`, `backward_kswap3`), 2.5 Priority-queue $k$-swap (`pq_kswap`), 2.6 Complexity, 2. Algorithms (+9 more)

### Community 5 - "Research notes — MILP, the beta constraint, and the road to the report"
Cohesion: 0.09
Nodes (21): 1. Where the project stands (results already in the repo), 2. To run on the server when it's back up, 3.1 What a MILP is, 3.2 How ecosystem pruning becomes a MILP (`epcrc/milp.py`), 3.3 The one thing the MILP canNOT encode — and why that's still useful, 3.4 The oracle–protocol gap is itself a finding, 3.5 Practical solver notes, 3. MILP, explained from scratch (+13 more)

### Community 6 - "milp_min_representative_set"
Cohesion: 0.18
Nodes (15): BackwardKSwapPruner, ForwardKSwapPruner, PriorityQueuePruner, Lazy greedy forward + backward cleanup, with optional k-swap escape.      Phase, Backward elimination seed + the k-swap reduction escape.      Backward eliminati, Forward selection seed + the k-swap reduction escape (no backward cleanup)., audit_disco(), audit_monotonicity() (+7 more)

### Community 7 - "test_task_error.py"
Cohesion: 0.19
Nodes (19): beta_coverage(), joint_feasible(), ndarray, quality_eligible_set(), Task-error preservation (the "beta" constraint).  The gamma constraint bounds *b, Models whose OWN task error on the shared eval sample is <= beta.      The quali, Feasibility under BOTH constraints: E(S) <= gamma and B(S) <= beta.      Drop-in, Per-model (err_orig, err_sub, delta) on the eval sample.      Models inside S su (+11 more)

### Community 8 - "__init__.py"
Cohesion: 0.21
Nodes (10): ABC, Intervention, ModelUnit, Any, Scalarizer, Ecosystem, Any, ndarray (+2 more)

### Community 9 - "coverage.py"
Cohesion: 0.14
Nodes (15): DISCOSolver, ndarray, Mask divide/overflow/invalid flags raised by the BLAS matmul kernel.      numpy, Solve min ||target - peers @ w||^2 s.t. w >= 0, sum(w) = 1 via SLSQP.      Only, Simplex-constrained projection utilities.      For target vector y and peer matr, _slsqp_simplex(), suppress_spurious_blas_flags(), errstate (+7 more)

### Community 10 - "PriorityQueuePruner"
Cohesion: 0.21
Nodes (7): PruningResult, PruningStep, Greedy backward sweep: remove any model whose removal keeps E(S) <= gamma., Remove-k-add-(k-1) local search to shrink |S| past the phase-2 local minimum., Trim redundant models, then package the result., Backward elimination that accepts a removal only when E_UCB <= gamma.      Stric, RiskControlledBackwardPruner

### Community 11 - "Math companion — every formula in the project, written out completely"
Cohesion: 0.25
Nodes (7): 1. The objects everything is built from, 2. The algorithms as formulas, 3. The MILP, written out completely, 4. Beta (task-error preservation), with full derivations, 5. Risk-controlled pruning (UCB), 6. Carathéodory: the geometry that predicts your numbers, Math companion — every formula in the project, written out completely

### Community 12 - "CoverageFunctional"
Cohesion: 0.16
Nodes (11): CoverageFunctional, ndarray, Compute per-model substitution errors U(i|S) and coverage E(S)., Detailed per-step metrics for coverage-based pruning.  Given a CoverageFunctiona, Compute a rich per-step metrics dict for a kept set S.      Returns a JSON-seria, step_metrics(), Sanity checks for the MILP oracle optimum and the k-swap reduction escape.  Reus, MILP size == brute-force protocol optimum on a small fit==eval instance.      Wi (+3 more)

### Community 15 - "experiment_0_warm_forward.py"
Cohesion: 0.22
Nodes (11): Warm-started forward pruning seeded with an initial kept set.      Starts from a, WarmStartForwardWorstCoveredPruner, aggregate(), farthest_pair_mean_abs(), main(), print_size_table(), ndarray, Task 3: UTD19 warm-forward gamma sweep with random subset runs.  Compares three (+3 more)

### Community 16 - "Callout"
Cohesion: 0.16
Nodes (3): Callout, FigBlock, PageBreak

### Community 17 - "Doc"
Cohesion: 0.22
Nodes (3): Doc, Heading, Flow blocks onto pages. Returns {heading label: page number}.

### Community 18 - "build_or_load_bundle"
Cohesion: 0.27
Nodes (11): DataFrame, build_or_load_bundle(), build_xy(), _load_raw_csv(), prepare_utd19(), ndarray, Shared UTD19 data loading, preprocessing, and per-city model training.  Trained, Load data, train per-city models, build Y_fit/Y_eval. Cache everything. (+3 more)

### Community 19 - "Para"
Cohesion: 0.24
Nodes (3): Para, _PreWrapped, Return (head, tail) blocks so that head fits in ``avail`` inches.

### Community 20 - "test_pruning_synthetic.py"
Cohesion: 0.25
Nodes (10): _build_cloud(), _coverage(), Task-1 sanity checks on a tiny synthetic cloud with a KNOWN answer.  Constructio, Every returned set must satisfy E(S) <= gamma., 3 triangle vertices + 5 strictly-interior points -> Y of shape (2, 8)., Backward must return EXACTLY the 3 hull vertices, with E <= gamma and no     fur, Forward's first feasible prefix must CONTAIN all 3 hull vertices, and be     fea, test_backward_returns_exact_hull_vertices() (+2 more)

### Community 21 - "doclib.py"
Cohesion: 0.24
Nodes (7): _emit_span(), _emit_words(), Minimal flowing-document engine on top of matplotlib's PDF backend.  Produces a, Split into (text, weight, style) tokens.      ``**bold**`` and ``__italic__`` ar, Emit words from ``chunk``, keeping ``$...$`` spans as single tokens., tokenize(), Contact-sheet renderer: draws every figure in figs.py to _png/ for review.

### Community 22 - "experiment_forward_beats_backward.py"
Cohesion: 0.29
Nodes (9): main(), mean_pairwise_dist(), pick_smallest(), plot_instance(), ndarray, Task 2: find a SEPARATING instance where forward < backward.  The headline findi, Smallest = fewest models N, then largest gap, then dim=2 (plottable)., pts: (N, d). Mean Euclidean distance between distinct points. (+1 more)

### Community 23 - "measure"
Cohesion: 0.25
Nodes (3): Eq, measure(), Width and height of a rendered string, in inches.

### Community 25 - "SubstitutionCertificate"
Cohesion: 0.38
Nodes (3): Sum of U(i|S) over all models i., Certificate for substituting one model using kept set S., SubstitutionCertificate

### Community 26 - "build.py"
Cohesion: 0.60
Nodes (4): build(), cover(), make_toc(), Build JUDGE_PANEL_GEOMETRY.pdf -- a visual math companion to the LLM judge-panel

## Knowledge Gaps
- **39 isolated node(s):** `graphify`, `EPCRC`, `Routing semantics`, `2.1 Forward selection (`forward`)`, `2.2 Forward selection with trimming (`forward_trim`)` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CoverageFunctional` connect `CoverageFunctional` to `BackwardEliminationPruner`, `CoverageFunctional`, `test_sparse_and_risk.py`, `milp_min_representative_set`, `test_task_error.py`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `experiment_0_warm_forward.py`, `test_pruning_synthetic.py`, `experiment_forward_beats_backward.py`, `SubstitutionCertificate`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `BackwardEliminationPruner` connect `BackwardEliminationPruner` to `CoverageFunctional`, `test_sparse_and_risk.py`, `milp_min_representative_set`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `CoverageFunctional`, `experiment_0_warm_forward.py`, `test_pruning_synthetic.py`, `experiment_forward_beats_backward.py`, `SubstitutionCertificate`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `ForwardSelectionPruner` connect `BackwardEliminationPruner` to `CoverageFunctional`, `milp_min_representative_set`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `CoverageFunctional`, `experiment_0_warm_forward.py`, `test_pruning_synthetic.py`, `experiment_forward_beats_backward.py`, `SubstitutionCertificate`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `CoverageFunctional` (e.g. with `DISCOSolver` and `BackwardEliminationPruner`) actually correct?**
  _`CoverageFunctional` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `BackwardEliminationPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`BackwardEliminationPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ForwardSelectionPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`ForwardSelectionPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `PriorityQueuePruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`PriorityQueuePruner` has 3 INFERRED edges - model-reasoned connections that need verification._