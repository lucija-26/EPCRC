# Graph Report - EPCRC  (2026-09-13)

## Corpus Check
- 136 files · ~1,096,975 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 983 nodes · 2137 edges · 62 communities (41 shown, 21 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 75 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8fc8bd96`
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
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_doclib.py|doclib.py]]
- [[_COMMUNITY_experiment_forward_beats_backward.py|experiment_forward_beats_backward.py]]
- [[_COMMUNITY_measure|measure]]
- [[_COMMUNITY_Block|Block]]
- [[_COMMUNITY_smoke_core8.py|smoke_core8.py]]
- [[_COMMUNITY_build.py|build.py]]
- [[_COMMUNITY_Space|Space]]
- [[_COMMUNITY_Bullet|Bullet]]
- [[_COMMUNITY_Context|Context]]
- [[_COMMUNITY_rewardbench.py|rewardbench.py]]
- [[_COMMUNITY_LabelScorer|LabelScorer]]
- [[_COMMUNITY_BackwardEliminationPruner|BackwardEliminationPruner]]
- [[_COMMUNITY_build_or_load_bundle|build_or_load_bundle]]
- [[_COMMUNITY_SubstitutionCertificate|SubstitutionCertificate]]
- [[_COMMUNITY_run_all.py|run_all.py]]
- [[_COMMUNITY_render|render]]
- [[_COMMUNITY_synthetic_judges.py|synthetic_judges.py]]
- [[_COMMUNITY_load_panel|load_panel]]
- [[_COMMUNITY_experiment_replay_forward_seeds.py|experiment_replay_forward_seeds.py]]
- [[_COMMUNITY_export_results.py|export_results.py]]
- [[_COMMUNITY_manifest.json|manifest.json]]
- [[_COMMUNITY_figures.py|figures.py]]
- [[_COMMUNITY_test_sparse_and_risk.py|test_sparse_and_risk.py]]
- [[_COMMUNITY_test_panel_registry.py|test_panel_registry.py]]
- [[_COMMUNITY_Core-20 server runbook|Core-20 server runbook]]
- [[_COMMUNITY_C3 — coverage-based selection beats the baselines|C3 — coverage-based selection beats the baselines]]
- [[_COMMUNITY_test_duplicate_judges_are_not_both_selected_early|test_duplicate_judges_are_not_both_selected_early]]
- [[_COMMUNITY_test_c2_full_panel_reconstructs_itself_exactly|test_c2_full_panel_reconstructs_itself_exactly]]
- [[_COMMUNITY_test_c3_headline_excludes_the_trivial_budgets|test_c3_headline_excludes_the_trivial_budgets]]
- [[_COMMUNITY_test_c3_headline_k_values_are_plain_ints|test_c3_headline_k_values_are_plain_ints]]
- [[_COMMUNITY_test_c3_exhaustive_selection_transfers_to_held_out_items|test_c3_exhaustive_selection_transfers_to_held_out_items]]
- [[_COMMUNITY_test_paired_table_covers_the_same_budgets_as_the_headline|test_paired_table_covers_the_same_budgets_as_the_headline]]
- [[_COMMUNITY_test_paired_outcome_labels_a_zero_difference_as_a_tie|test_paired_outcome_labels_a_zero_difference_as_a_tie]]
- [[_COMMUNITY_test_paired_is_more_sensitive_than_comparing_two_intervals|test_paired_is_more_sensitive_than_comparing_two_intervals]]
- [[_COMMUNITY_test_split_robustness_reports_min_sd_and_max_not_just_the_mean|test_split_robustness_reports_min_sd_and_max_not_just_the_mean]]
- [[_COMMUNITY_test_split_robustness_agrees_with_the_headline_mean|test_split_robustness_agrees_with_the_headline_mean]]
- [[_COMMUNITY_test_simplex_reconstruction_never_leaves_the_simplex|test_simplex_reconstruction_never_leaves_the_simplex]]
- [[_COMMUNITY_test_every_reconstruction_rule_sees_the_same_kept_judges|test_every_reconstruction_rule_sees_the_same_kept_judges]]
- [[_COMMUNITY_test_manifest_hashes_match_the_files_on_disk|test_manifest_hashes_match_the_files_on_disk]]
- [[_COMMUNITY_test_summary_reports_the_headline_numbers_it_claims|test_summary_reports_the_headline_numbers_it_claims]]
- [[_COMMUNITY_test_c1_breakpoints_are_where_the_removable_set_changes|test_c1_breakpoints_are_where_the_removable_set_changes]]

## God Nodes (most connected - your core abstractions)
1. `CoverageFunctional` - 72 edges
2. `JudgeResponses` - 47 edges
3. `BackwardEliminationPruner` - 33 edges
4. `ForwardSelectionPruner` - 28 edges
5. `JudgeCoverageFunctional` - 27 edges
6. `Panel` - 26 edges
7. `panels()` - 21 edges
8. `PriorityQueuePruner` - 21 edges
9. `SubstitutionCertificate` - 20 edges
10. `BackwardKSwapPruner` - 20 edges

## Surprising Connections (you probably didn't know these)
- `ExactSearcher` --uses--> `CoverageFunctional`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/coverage.py
- `test_panel_weight_is_the_bf16_total()` --calls--> `panel_weight_gb()`  [EXTRACTED]
  tests/test_panel_registry.py → epcrc/panel.py
- `test_resplitting_keeps_the_judges_and_the_item_total()` --calls--> `load_panel()`  [EXTRACTED]
  tests/test_panel_split_seeds.py → epcrc/panel.py
- `test_split_seed_defaults_to_the_pairs_seed()` --calls--> `load_panel()`  [EXTRACTED]
  tests/test_panel_split_seeds.py → epcrc/panel.py
- `ExactSearcher` --uses--> `ForwardSelectionPruner`  [INFERRED]
  experiments/experiment_exact_optimum.py → epcrc/pruning.py

## Import Cycles
- None detected.

## Communities (62 total, 21 thin omitted)

### Community 0 - "BackwardEliminationPruner"
Cohesion: 0.14
Nodes (25): BackwardEliminationPruner, ForwardKSwapPruner, ForwardSelectionPruner, PriorityQueuePruner, Section 4.1 backward elimination.      Start with S = J and remove one model at, Section 4.2 forward selection.      Start with S = empty and greedily add the mo, Lazy greedy forward + backward cleanup, with optional k-swap escape.      Phase, Forward selection seed + the k-swap reduction escape (no backward cleanup). (+17 more)

### Community 1 - "figs.py"
Cohesion: 0.12
Nodes (55): cloud_panel(), main(), One figure: how EPCRC and the LLM judge panel are the same problem.  Writes docs, Draw a row of coordinate cells; group cells into blocks of `group`., vector_strip(), bare_ax(), _ccw(), _circle_judges() (+47 more)

### Community 2 - "CoverageFunctional"
Cohesion: 0.09
Nodes (45): Panel, ndarray, The scored panel, split into FIT / CERT / TEST along the item axis.      ``split, build_selectors(), Every competitor in the C3 comparison, keyed by name., Section 21.1 'cheapest models first'., Section 21.1 'largest models first', using measured cost as the size proxy., select_cost_ascending() (+37 more)

### Community 3 - "test_sparse_and_risk.py"
Cohesion: 0.13
Nodes (28): base_item_id(), build_pairs(), grouped_split(), is_tie_row(), Turn base tasks into at most `max_pairs` judged pairs each.      Non-tie tasks p, Assign base tasks to FIT / CERT / TEST, stratified and grouped.      Splitting h, Grouping key. Plain `id` collides across subsets, so qualify it., normalize_label_scores() (+20 more)

### Community 4 - "2. Algorithms"
Cohesion: 0.11
Nodes (17): 1. Problem statement, 2.1 Forward selection (`forward`), 2.2 Forward selection with trimming (`forward_trim`), 2.3 Backward elimination (`backward`), 2.4 $k$-swap (`backward_kswap2`, `backward_kswap3`), 2.5 Priority-queue $k$-swap (`pq_kswap`), 2.6 Complexity, 2. Algorithms (+9 more)

### Community 5 - "Research notes — MILP, the beta constraint, and the road to the report"
Cohesion: 0.09
Nodes (21): 1. Where the project stands (results already in the repo), 2. To run on the server when it's back up, 3.1 What a MILP is, 3.2 How ecosystem pruning becomes a MILP (`epcrc/milp.py`), 3.3 The one thing the MILP canNOT encode — and why that's still useful, 3.4 The oracle–protocol gap is itself a finding, 3.5 Practical solver notes, 3. MILP, explained from scratch (+13 more)

### Community 6 - "milp_min_representative_set"
Cohesion: 0.12
Nodes (29): load_panel(), Rebuild the scored judge panel from the cached (judge, context) blocks.  Two see, Rebuild the response tensor, optionally re-splitting it under another seed., apply_weights(), bootstrap(), _evaluate_subset(), fit_panel_weights(), _group_totals() (+21 more)

### Community 7 - "test_task_error.py"
Cohesion: 0.10
Nodes (29): ABC, Intervention, ModelUnit, Any, Scalarizer, Ecosystem, Any, ndarray (+21 more)

### Community 8 - "__init__.py"
Cohesion: 0.17
Nodes (16): context_errors(), ndarray, Three-class judge outputs: total-variation loss and worst-context fitting.  An L, Total variation between three-class rows, along the last axis., Mean TV error of the reconstruction of `target_idx`, one value per context., Fit one simplex weight vector that minimises the worst context's mean TV.      S, solve_minimax_weights(), total_variation() (+8 more)

### Community 9 - "coverage.py"
Cohesion: 0.14
Nodes (15): DISCOSolver, ndarray, Mask divide/overflow/invalid flags raised by the BLAS matmul kernel.      numpy, Solve min ||target - peers @ w||^2 s.t. w >= 0, sum(w) = 1 via SLSQP.      Only, Simplex-constrained projection utilities.      For target vector y and peer matr, _slsqp_simplex(), suppress_spurious_blas_flags(), errstate (+7 more)

### Community 10 - "PriorityQueuePruner"
Cohesion: 0.10
Nodes (31): _cert_check(), _cert_init(), CertificateResult, certify_lower_bound(), is_feasible_set(), milp_min_representative_set(), MilpResult, min_substitution_error() (+23 more)

### Community 11 - "Math companion — every formula in the project, written out completely"
Cohesion: 0.25
Nodes (7): 1. The objects everything is built from, 2. The algorithms as formulas, 3. The MILP, written out completely, 4. Beta (task-error preservation), with full derivations, 5. Risk-controlled pruning (UCB), 6. Carathéodory: the geometry that predicts your numbers, Math companion — every formula in the project, written out completely

### Community 12 - "CoverageFunctional"
Cohesion: 0.09
Nodes (19): CoverageFunctional, ndarray, Sum of U(i|S) over all models i., Certificate for substituting one model using kept set S., Compute per-model substitution errors U(i|S) and coverage E(S)., SubstitutionCertificate, Detailed per-step metrics for coverage-based pruning.  Given a CoverageFunctiona, Compute a rich per-step metrics dict for a kept set S.      Returns a JSON-seria (+11 more)

### Community 15 - "experiment_0_warm_forward.py"
Cohesion: 0.20
Nodes (11): Warm-started forward pruning seeded with an initial kept set.      Starts from a, WarmStartForwardWorstCoveredPruner, aggregate(), farthest_pair_mean_abs(), main(), print_size_table(), ndarray, Task 3: UTD19 warm-forward gamma sweep with random subset runs.  Compares three (+3 more)

### Community 16 - "Callout"
Cohesion: 0.16
Nodes (3): Callout, FigBlock, PageBreak

### Community 17 - "Doc"
Cohesion: 0.22
Nodes (3): Doc, Heading, Flow blocks onto pages. Returns {heading label: page number}.

### Community 18 - "build_or_load_bundle"
Cohesion: 0.06
Nodes (55): JudgeCoverageFunctional, Coverage functional for three-class judge panels under worst-context TV.      De, curved_arc(), duplicated_extremes(), _panel(), ndarray, _random_interior(), Controlled judge panels whose correct answer is known before running anything. (+47 more)

### Community 19 - "Para"
Cohesion: 0.24
Nodes (3): Para, _PreWrapped, Return (head, tail) blocks so that head fits in ``avail`` inches.

### Community 20 - "__init__.py"
Cohesion: 0.24
Nodes (9): JudgedPair, One A-versus-B comparison put to a judge.      `gold_label` is "A", "B" or "C";, accuracy_report(), argmax_labels(), Deterministic three-label probability extraction from an open-weight judge.  The, Sanity statistics for one judge under one context.      A judge whose prediction, Per-model record of how the three labels encode., TokenizationReport (+1 more)

### Community 21 - "doclib.py"
Cohesion: 0.24
Nodes (7): _emit_span(), _emit_words(), Minimal flowing-document engine on top of matplotlib's PDF backend.  Produces a, Split into (text, weight, style) tokens.      ``**bold**`` and ``__italic__`` ar, Emit words from ``chunk``, keeping ``$...$`` spans as single tokens., tokenize(), Contact-sheet renderer: draws every figure in figs.py to _png/ for review.

### Community 22 - "experiment_forward_beats_backward.py"
Cohesion: 0.29
Nodes (9): main(), mean_pairwise_dist(), pick_smallest(), plot_instance(), ndarray, Task 2: find a SEPARATING instance where forward < backward.  The headline findi, Smallest = fewest models N, then largest gap, then dim=2 (plottable)., pts: (N, d). Mean Euclidean distance between distinct points. (+1 more)

### Community 23 - "measure"
Cohesion: 0.25
Nodes (3): Eq, measure(), Width and height of a rendered string, in inches.

### Community 25 - "smoke_core8.py"
Cohesion: 0.16
Nodes (26): panel_weight_gb(), Approximate bf16 download size of a set of judges, in GB., A deterministic domain-balanced subset, for smoke runs and audits., read_pairs(), stratified_subset(), _cache_path(), _check_model_access(), configure() (+18 more)

### Community 26 - "build.py"
Cohesion: 0.60
Nodes (4): build(), cover(), make_toc(), Build JUDGE_PANEL_GEOMETRY.pdf -- a visual math companion to the LLM judge-panel

### Community 29 - "Context"
Cohesion: 0.12
Nodes (20): canonical_gold_label(), canonicalize(), Context, prompt_sha256(), protocol_hashes(), Frozen judge prompt protocols, interventions, and prompt rendering.  The retaine, One registered evaluation condition.      `swap` exchanges the two responses, wh, Render one prompt under a context.      Interventions are applied to the *conten (+12 more)

### Community 30 - "rewardbench.py"
Cohesion: 0.20
Nodes (13): _digest(), load_reward_bench_2(), RewardBench 2 -> judged pairs, and the grouped stratified split.  Verified again, Checksummed record of one split, so results can be traced to their data., One JSON object per line, so the scorer can stream it., Stable 64-bit integer from the string form of `parts`., Load the raw dataset. Pin `revision` before formal inference., Indices of the `k` candidates whose (seed, key, content) hash is smallest. (+5 more)

### Community 31 - "LabelScorer"
Cohesion: 0.17
Nodes (10): LabelScorer, ndarray, Record how each label encodes, and whether the labels are comparable., Wrap the prompt in the model's official chat template.          Qwen 3 exposes a, Three-class probabilities for one rendered prompt., Three-class probabilities for a batch of rendered prompts., Exact sequence log-likelihood of a multi-token label., Canonical three-class probabilities for every pair under one context.      Retur (+2 more)

### Community 32 - "BackwardEliminationPruner"
Cohesion: 0.25
Nodes (10): _build_cloud(), _coverage(), Task-1 sanity checks on a tiny synthetic cloud with a KNOWN answer.  Constructio, Every returned set must satisfy E(S) <= gamma., 3 triangle vertices + 5 strictly-interior points -> Y of shape (2, 8)., Backward must return EXACTLY the 3 hull vertices, with E <= gamma and no     fur, Forward's first feasible prefix must CONTAIN all 3 hull vertices, and be     fea, test_backward_returns_exact_hull_vertices() (+2 more)

### Community 33 - "build_or_load_bundle"
Cohesion: 0.27
Nodes (11): build_or_load_bundle(), build_xy(), _load_raw_csv(), prepare_utd19(), DataFrame, ndarray, Shared UTD19 data loading, preprocessing, and per-city model training.  Trained, Load data, train per-city models, build Y_fit/Y_eval. Cache everything. (+3 more)

### Community 34 - "SubstitutionCertificate"
Cohesion: 0.11
Nodes (25): JudgeResponses, Three-class judge probabilities grouped by evaluation context.      ``blocks[c]`, Total number of items across all contexts., _chain_from_labels(), judge_matrix(), _medoid(), pairwise_tv(), ndarray (+17 more)

### Community 36 - "render"
Cohesion: 0.29
Nodes (5): PruningResult, PruningStep, Greedy backward sweep: remove any model whose removal keeps E(S) <= gamma., Remove-k-add-(k-1) local search to shrink |S| past the phase-2 local minimum., Trim redundant models, then package the result.

### Community 37 - "synthetic_judges.py"
Cohesion: 0.14
Nodes (28): fit_weights(), ndarray, Error of a fitted weight vector on a held-out split.      ``off_simplex_frac`` i, Fit one weight vector over `kept` that reconstructs `target` on FIT., score_weights(), lowrank_floor(), Rank-k reconstruction error using any subspace, not just real judges.      This, Properties the C3 comparison depends on.  C3 is an equal-k, equal-cost head-to-h (+20 more)

### Community 38 - "load_panel"
Cohesion: 0.31
Nodes (8): The split seeds must re-partition one fixed item universe.  E1 needs confidence, Otherwise the 'bands' would be five copies of one number., _splits(), test_every_split_seed_covers_the_same_item_universe(), test_resplitting_keeps_the_judges_and_the_item_total(), test_split_seed_defaults_to_the_pairs_seed(), test_split_seeds_actually_disagree_about_membership(), test_splits_are_disjoint()

### Community 39 - "experiment_replay_forward_seeds.py"
Cohesion: 0.13
Nodes (27): c1_headline(), c1_table(), c2_table(), c3_cost_table(), c3_headline(), c3_paired_table(), c3_split_robustness(), c3_table() (+19 more)

### Community 40 - "export_results.py"
Cohesion: 0.15
Nodes (23): build_summary(), build_tables(), _c1_section(), _c2_section(), _c3_paired_section(), _c3_robustness_section(), _c3_section(), copy_notebooks() (+15 more)

### Community 41 - "manifest.json"
Cohesion: 0.08
Nodes (23): path, present, path, present, path, present, path, present (+15 more)

### Community 42 - "figures.py"
Cohesion: 0.17
Nodes (20): fig_c1_real(), fig_c1_synthetic(), fig_c2_frontier(), fig_c3_cost(), fig_c3_headline(), fig_c3_paired(), fig_c3_reconstruction(), fig_c3_split_robustness() (+12 more)

### Community 43 - "test_sparse_and_risk.py"
Cohesion: 0.16
Nodes (17): coverage_ucb(), Risk-controlled pruning (paper Open Problem 1 / section 4.3, Experiment 4).  Poi, Per-model UCB on U(i|S), union-bounded over the N models., E_UCB(S) = max_i UCB(i|S)., Backward elimination that accepts a removal only when E_UCB <= gamma.      Stric, RiskControlledBackwardPruner, uniqueness_ucb(), _noisy_instance() (+9 more)

### Community 44 - "test_panel_registry.py"
Cohesion: 0.12
Nodes (9): Where a panel's cached (judge, context) blocks live., scores_dir(), The panel registry must match plan section 10 exactly.  A typo in a judge id or, A Core-20 run must not overwrite the Core-8 blocks that licensed it., C1 needs judges that could plausibly be redundant given a sibling.      Core-8 d, test_core20_has_within_family_pairs(), test_panel_weight_is_the_bf16_total(), test_panels_write_to_separate_directories() (+1 more)

### Community 45 - "Core-20 server runbook"
Cohesion: 0.15
Nodes (12): 0. Before starting the job, 1. Start the job, 2. Set up the environment, 3. Gate G0 — fail cheaply, 4. Gates G1 and G2, 5. Score the panel, 6. Run the three claims, 7. Build the package (+4 more)

### Community 46 - "C3 — coverage-based selection beats the baselines"
Cohesion: 0.15
Nodes (12): C1 — individual redundancy certificates do not compose, C2 — the physical-to-virtual compression frontier, C3 — coverage-based selection beats the baselines, Cost, Does selection on FIT transfer to TEST?, Does the result depend on the split?, EPCRC results — core8, Head-to-head on the same items (+4 more)

## Knowledge Gaps
- **78 isolated node(s):** `generated_utc`, `panel`, `git_commit`, `git_branch`, `git_clean` (+73 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CoverageFunctional` connect `CoverageFunctional` to `BackwardEliminationPruner`, `BackwardEliminationPruner`, `SubstitutionCertificate`, `render`, `test_task_error.py`, `__init__.py`, `coverage.py`, `PriorityQueuePruner`, `test_sparse_and_risk.py`, `experiment_0_warm_forward.py`, `build_or_load_bundle`, `experiment_forward_beats_backward.py`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `JudgeResponses` connect `SubstitutionCertificate` to `CoverageFunctional`, `synthetic_judges.py`, `milp_min_representative_set`, `test_task_error.py`, `__init__.py`, `CoverageFunctional`, `test_duplicate_judges_are_not_both_selected_early`, `build_or_load_bundle`, `smoke_core8.py`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `JudgeCoverageFunctional` connect `build_or_load_bundle` to `CoverageFunctional`, `test_task_error.py`, `__init__.py`, `CoverageFunctional`, `smoke_core8.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `CoverageFunctional` (e.g. with `DISCOSolver` and `JudgeCoverageFunctional`) actually correct?**
  _`CoverageFunctional` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `JudgeResponses` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`JudgeResponses` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `BackwardEliminationPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`BackwardEliminationPruner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ForwardSelectionPruner` (e.g. with `CoverageFunctional` and `SubstitutionCertificate`) actually correct?**
  _`ForwardSelectionPruner` has 3 INFERRED edges - model-reasoned connections that need verification._