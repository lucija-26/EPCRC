# Research notes — MILP, the beta constraint, and the road to the report

*Written 2026-07-02 (Claude Fable session). Everything here is self-contained:
read it alongside the code, no chat history needed.*

---

## 1. Where the project stands (results already in the repo)

- `E(S) = max_i U(i|S)` is a max of residuals → **not submodular**, so the
  paper's §4.2 intuition ("forward usually yields smaller sets") fails here.
  Backward elimination strongly beats forward on UTD19 and synthetic data.
- **Backward + k-swap is (empirically) exactly optimal.**
  `results/exact_optimum/exact_optimum_3seeds.json` (3 seeds × subset 12,
  γ = 60…160, protocol-true exhaustive optimum):
  - `backward_kswap3`: optimum on **18/18** instances.
  - `backward_kswap2`: optimum on 17/18 (one +1 miss at γ=80).
  - plain backward: gap ≈ +0.7; forward: up to **+4.7** (γ=80).
  - Synthetic replay study: bwd+kswap(k=2) optimal on 99% of random
    instances; its only failures are optima sharing **zero** models with the
    backward seed (exactly the remove-3/add-2 move), and k=3 fixed all of them.
- **Why k=2 → k=3 matters**: k=1 reduction ≡ a backward step. k=2
  (remove 2, add 1) is the minimal escape from backward's local optimum
  (monotonicity forbids removing ≥2 without adding ≥1 back — see the
  `BackwardKSwapPruner` docstring). k=2 fails only when the optimum is
  disjoint from the seed; k=3 covers that. Cost per sweep is
  `C(|S|,k)·C(N−|S|,k−1)` coverage evaluations — fine for N ≤ 20.
- **MILP is built and validated** (`epcrc/milp.py`, 5/5 vs brute force) —
  see §3 below for what it means and how to use it.
- First MILP numbers on UTD19 (subset 12, seed 0): γ=140 → |S|=1 (26 s),
  γ=100 → |S|=2 (33 s), γ=60 → |S|=4 (151 s). Compare protocol optimum
  mean 8.7 at γ=60: the **oracle–protocol gap is large** (see §3.4).

## 2. To run on the server when it's back up

```bash
git pull
# 1. protocol-true optimum, full subset size (hours; the main event)
python experiments/experiment_exact_optimum.py 10 "60,80,100,120,140,160" 15
# 2. MILP lower bound on the SAME subsets (joins per record)
python experiments/experiment_milp_optimum.py 10 "60,80,100,120,140,160" 15
# 3. MILP on the FULL 20-model ecosystem (enumeration can't reach this)
python experiments/experiment_milp_optimum.py 1 "60,80,100,120,140,160" 20
# 4. synthetic gap study, big sample
python experiments/experiment_synthetic_optimum.py 1000 "2,3"
# 5. (for beta, §4) regenerate the bundle so it contains ground truth:
#    delete data/exp_0_utd19_cache/bundle.npz and rerun the gamma-sweep /
#    pipeline entry point once; the new bundle gains y_fit_true / y_eval_true.
git add results/ && git commit -m "server results" && git push
```

Then locally: `git pull` and re-run `notebooks/optimality_gap_analysis.ipynb`
— it picks up whatever JSONs exist.

---

## 3. MILP, explained from scratch

### 3.1 What a MILP is

A **Mixed-Integer Linear Program** is an optimization problem where the
objective and all constraints are *linear*, but some variables are restricted
to integers (here: binary 0/1). Linear programs (LPs) are solvable in
polynomial time; adding integrality makes the problem NP-hard in general —
but modern solvers (Gurobi, HiGHS, CBC) solve surprisingly large instances
*to proven optimality* using **branch and bound**:

1. Solve the **LP relaxation** (let each binary z ∈ [0,1] be fractional).
   Its optimal value is a *lower bound* on the true optimum (minimization).
2. If some z is fractional (say z=0.4), **branch**: solve two subproblems
   with z=0 and z=1. Each inherits the parent's bound.
3. Any all-integer solution found is an *upper bound* (an incumbent).
4. **Prune** any branch whose LP bound is already worse than the incumbent.
5. When bounds meet, the incumbent is **provably optimal** — the solver
   reports a "MIP gap" of 0. You can also stop early and keep the gap as a
   certified interval around the optimum.

That last point is the whole appeal: unlike greedy or k-swap, a MILP returns
either the optimum *with a proof*, or a solution *plus a bound on how far off
it can be*.

### 3.2 How ecosystem pruning becomes a MILP (`epcrc/milp.py`)

Decision variables:

- `z_j ∈ {0,1}` — keep model j (this is what we minimize: `min Σ_j z_j`).
- `w_ij ≥ 0` — routing weight of target model i on model j.
- `e_ti ≥ 0` — envelope for the absolute residual of target i at eval row t.

Constraints, all linear:

| constraint | meaning |
|---|---|
| `Σ_j w_ij = 1` | weights live on the simplex |
| `w_ij ≤ z_j` | **the coupling trick**: you may only route to kept models. If z_j=0, every w_·j is forced to 0. (Works because w ≤ 1 on the simplex — a "big-M" with M=1.) |
| `e_ti ≥ ±(y_ti − (Y_eval w_i)_t)` | **absolute-value linearization**: e ≥ r and e ≥ −r together mean e ≥ \|r\| |
| `(1/n) Σ_t e_ti ≤ γ` | mean-abs substitution error of target i is within tolerance |

Note the max over i in `E(S) = max_i U(i|S) ≤ γ` costs nothing: "max ≤ γ" is
just "≤ γ for every i", one row per target. The same works for the `max`
metric (`e_ti ≤ γ` for all t); **RMSE would need a quadratic constraint**
(MIQCP — Gurobi handles it, scipy/HiGHS does not).

### 3.3 The one thing the MILP canNOT encode — and why that's still useful

The protocol in the paper (and everywhere in `epcrc/`) is *honest*: weights
are fit on `Y_fit` by least squares, then *scored* on `Y_eval`. "w is the
argmin of the fit problem" is a **bilevel** condition — encoding it exactly
needs the inner QP's KKT conditions as complementarity constraints, which is
technically possible (indicator constraints / big-M) but fragile and slow.

Instead the MILP answers the *existential* question: **does any simplex
router within S achieve eval error ≤ γ?** Call its optimum the **oracle
optimum**. Every protocol-feasible S is oracle-feasible (its fit weights are
a witness), so:

```
|OPT_oracle (MILP)|  ≤  |OPT_protocol (exhaustive)|  ≤  |backward+kswap|
```

You now have the optimum **sandwiched** between a certified lower bound that
scales to any N, and an algorithm that empirically sits at the upper end of
the sandwich touching the middle term.

### 3.4 The oracle–protocol gap is itself a finding

On UTD19 (subset 12, γ=60) the oracle optimum was **4** while the protocol
optimum averaged **8.7**. Interpretation: certificates that would substitute
everything with 4 models *exist*, but DISCO's honest fit-then-evaluate
procedure cannot find them — the price of honest splitting. Two readings for
the report:

- **Statistical**: the oracle router is allowed to "overfit" the eval sample;
  the gap partially reflects generalization, not just inefficiency. (A clean
  check: score the oracle-MILP weights on a *third* held-out sample.)
- **Governance**: if you must certify on held-out data (you do), the
  protocol optimum is the honest target, and backward+k-swap reaches it.

### 3.5 Practical solver notes

- Locally: `scipy.optimize.milp` (HiGHS) — what `epcrc/milp.py` uses. No
  license, no install beyond scipy.
- Server: DTU has **Gurobi academic licenses** — worth porting for the N=20+
  runs (the matrix layout is documented in `epcrc/milp.py`; Gurobi's Python
  API makes the port ~40 lines). Expect ~10× speedup.
- Runtime grows sharply as γ decreases (larger optimum ⇒ weaker LP bound ⇒
  more branching). Use `time_limit` and report the MIP gap when hit.
- Status codes from scipy: 0 = optimal, 1 = iteration/time limit (gap in
  `mip_gap`), 2 = infeasible.

---

## 4. The beta constraint (task-error preservation)

### 4.1 The idea, sharpened

γ says: *the router must mimic the removed model's outputs.* It never looks
at ground truth. Your β says: *after substitution, the system must still
predict actual traffic (about) as well as before.* Formally, with certificate
router `h_i = Y_eval[:, S] @ w_i(S)` and task loss L against ground truth y*:

```
delta(i|S) = L(h_i, y*) − L(f_i, y*)          task-error change
B(S)       = max_i delta(i|S)                  worst degradation
constraint:  B(S) ≤ β                           (jointly with E(S) ≤ γ)
```

This is implemented in `epcrc/task_error.py` (`substitution_task_errors`,
`beta_coverage`, `joint_feasible`) with tests in `tests/test_task_error.py`.

### 4.2 Two theorems that make β (mostly) free — verified, they're correct

**Bound 1 — coverage already limits quality drift (triangle inequality).**
For any norm-type loss L computed on the same sample and in the same metric
as the fidelity term,

    L(substitute, y*) ≤ L(f_i, y*) + L(substitute, f_i)  ≤  err_orig(i) + γ.

So `delta(i|S) ≤ γ` automatically: a substitute can never be more than γ
worse than the model it replaces. A separate β for *drift* only binds when
β < γ, when the task metric differs from the fidelity metric (MAE fidelity
does NOT bound RMSE drift — align them, don't mix), or in the relative form
`delta_i / err_orig_i ≤ β` which γ cannot express.

**Bound 2 — a convex blend is at least as good as its worst ingredient
(Jensen).** For convex L (MSE, MAE, RMSE after the root) and simplex weights,
per query and summed:

    L(Σ_j w_j Y_j, y*) ≤ Σ_j w_j L(Y_j, y*) ≤ max_{j∈S} L(Y_j, y*).

This holds for ANY simplex w — fit weights, oracle-MILP weights, anything —
and on any sample, so it survives the honest split. Usually strict:
averaging uncorrelated errors cancels them (the ensembling effect; see
`test_delta_can_be_negative` — delta is often *negative*).

**The design move (recommended): quality by construction.** Pre-filter the
keep-eligible pool to `K = {j : L(Y_j, y*) ≤ β}`, then run the ordinary
γ-pruning with S ⊆ K but still covering ALL of J:

    min |S|   s.t.   S ⊆ K,   max_{i∈J} U(i|S) ≤ γ.

By Bound 2 every certificate router is then automatically ≤ β on the task —
no constraint inside the loop, and (crucially) **the monotone coverage theory
is untouched**, unlike a joint (γ, B(S) ≤ β) test, since B(S) is not provably
monotone. Implemented as `quality_eligible_set` in `epcrc/task_error.py`.
All pruners and the MILP adapt trivially (restrict removals'/additions'
candidate pool; in the MILP fix `z_j = 0` for j ∉ K). One honest new
outcome becomes possible: **infeasibility** — if a hull archetype fails the
quality bar, no S ⊆ K reaches γ. That is a reportable governance finding
("this ecosystem cannot be consolidated to quality-β representatives at
tolerance γ"), not a bug.

### 4.3 How to choose the number β

Never an absolute number in flow units — anchor it, and keep β and γ in the
same metric/units (RMSE in veh/h is the natural choice; consider aligning the
fidelity metric to RMSE too):

1. **Relative to the replaced model**: `β_i = (1+ε)·err_orig(i)` — honest,
   but inherits each original's badness (a bad model licenses a bad
   substitute).
2. **Relative to a naive baseline**: β = RMSE of a persistence forecaster
   (predict next = current). Standard ML anchoring; any model worth keeping
   beats it, and by Bound 2 so does every substitute.
3. **Relative to the ecosystem itself**: β = max (or a high percentile) of
   the kept models' task error — "no substitute worse than the worst real
   model." **Caveat**: this needs each model's error on the SHARED POOLED
   eval sample; the cached `per_city_rmse` is *training* RMSE on the model's
   *own* city and is NOT the right quantity. The regenerated bundle
   (`y_eval_true`) makes the right one computable.
4. **Sweep it**: plot |S| vs (γ, β) as a 2-D frontier — quantifies exactly
   when β binds vs when Bounds 1–2 make it redundant. A genuinely new plot
   for this problem.

The framing for the report: γ is the *audit* constraint (label-free,
monotone, certifiable), β is the *deployment* constraint (needs labels).
Bounds 1–2 say β is nearly free; the pre-filter design makes it exactly free
inside the optimization. `substitution_task_errors` then reports the
*measured* deltas, which are typically far below the bounds (often < 0).

### 4.4 What's needed to run it on UTD19

The old cached `bundle.npz` lacks ground truth. The pipeline now saves
`y_fit_true` / `y_eval_true` (fixed in `epcrc/utd19_pipeline.py`) — on the
server, delete `data/exp_0_utd19_cache/bundle.npz` and rerun any experiment
that builds the bundle. **Caveat**: regenerating resamples the shared query
points, so γ-only results will shift slightly; either rerun the γ sweeps on
the new bundle or keep old/new results clearly separated.

Then a beta experiment is ~30 lines: sweep β ∈ {0, 0.01, 0.05, 0.10}
(relative) × γ grid, using `joint_feasible` as the acceptance test inside a
backward/k-swap loop, and record |S| and which constraint was binding.

### 4.5 β in the MILP

With mean-abs task loss, the β constraint is *linear* in w (same
absolute-value trick against y* instead of against model i), so it drops
straight into `epcrc/milp.py` as extra rows. With MSE it's convex quadratic —
fine for Gurobi (MIQCP), not for HiGHS/scipy. The relative form stays linear
(err_orig_i is a constant).

---

## 4.9 Scorecard: the paper's open problems & experiments vs this repo

| paper item | status | where |
|---|---|---|
| OP1 certifiable coverage (UCB) | **code ready, not run** | `epcrc/risk.py` (`RiskControlledBackwardPruner`); union bound over models, per fixed S; adaptive-uniform bounds remain the open problem — say so honestly |
| OP2 complexity of minimal sets | **answered empirically** | exact optimum + MILP + k-swap gap tables; NP-hardness proof still open (fine to state as conjecture) |
| OP3 active query design | not touched | out of scope for the special course; one sentence in the report |
| OP4 robust pruning across contexts | not touched | recipe: split UTD19 eval queries into contexts (e.g. per-city pools or time-of-day), E_rob(S)=max_c E_c(S); all pruners work unchanged on E_rob |
| OP5 sparse certificates | **code ready, not run** | `milp_min_representative_set(..., max_support=r)`; sweep r = 1,2,3,∞ → the size-vs-sparsity tradeoff plot (paper eq. 11) |
| Exp 1 consolidation curves | done | gamma sweep experiments |
| Exp 2 robust contexts | not done | = OP4 recipe above |
| Exp 3 sparse certificates | **unlocked** | sparse MILP sweep, ~10 lines around `experiment_milp_optimum.py` |
| Exp 4 risk-controlled pruning | **unlocked** | compare `BackwardEliminationPruner` vs `RiskControlledBackwardPruner` across γ and δ; report size premium + violation rates under resampling |
| Exp 5 synthetic scaling | done | replay + synthetic optimum studies |

## 5. Open threads, in priority order

1. **Server runs** (§2) → final tables → write the optimality-gap section.
2. **Price of honest splitting** (§3.4): add the third-sample check for the
   oracle weights; this decides between the two interpretations.
3. **β frontier** (§4.3 option 3) after the bundle regeneration.
4. **A conjecture for the professor**: bwd+k-swap needed k=3 exactly when the
   optimum was disjoint from the backward seed. Is there an instance where
   k=3 fails (optimum "doubly disjoint")? Constructing one, or failing
   convincingly, are both good outcomes. Suspicion: needed k relates to the
   number of hull archetypes / intrinsic dimension.
5. **Gurobi port** of `epcrc/milp.py` for N ≥ 20 and MIQCP variants (β with
   MSE, RMSE fidelity).

## 6. Report skeleton (suggestion)

1. Problem + protocol recap (paper §2, your notation).
2. Why forward greedy fails here: non-submodularity + no trimming; the +4.7
   gap figure.
3. Backward + k-swap: the algorithm, the monotonicity argument for why k=2
   is the minimal escape, k=3 completeness on all tested instances.
4. Exact optimum: exhaustive (protocol) and MILP (oracle) — the sandwich.
5. The price of honest splitting.
6. β: definition, one-sidedness, the (γ, β) frontier.
7. Sparse certificates (size vs r) and risk-controlled pruning (size premium
   for certification) — the two paper experiments this repo now unlocks.
8. Open problems revisited (which of the paper's OP1–OP5 you touched).

## 7. One-month plan (server back, no Fable)

Week 1 — runs + core tables. `git pull`, run §2 commands, plus:
```bash
# sparse MILP sweep (Exp 3): edit experiment_milp_optimum.py to loop
#   max_support in [1, 2, 3, None]  around the milp_min_representative_set call
# risk-controlled comparison (Exp 4): backward vs RiskControlledBackwardPruner
#   on the UTD19 bundle, gammas 60..160, delta in [0.01, 0.05, 0.1]
# beta: delete data/exp_0_utd19_cache/bundle.npz, rerun a sweep to regenerate
#   (now with y_eval_true), then quality_eligible_set + backward+kswap on the pool
```
Push the JSONs, re-run `notebooks/optimality_gap_analysis.ipynb` locally.

Week 2 — the two new plots: size-vs-r (sparse) and size-premium-vs-δ
(risk-controlled); plus the (γ, β) frontier if the bundle regen went well.

Weeks 3–4 — write the report along §6. Every number you need is then in
`results/`. If time remains: the k=3-failure-instance hunt (§5 item 4), or
OP4 contexts.

## 8. Working with Claude (Opus) after this session

Opus in Claude Code is fully capable of continuing this — the quality of the
continuation depends far more on the context it gets than on the model. What
survives this session automatically:

- **This file** — tell it: *"Read docs/RESEARCH_NOTES.md and CLAUDE.md first,
  then <task>."* That one sentence replaces most of the shared history.
- **Project memory** (`MEMORY.md` in the Claude project dir) — loaded
  automatically each session on this machine; it summarizes the findings.
- **The tests** (`pytest tests/ -q`, 14 passing) — insist any change keeps
  them green; they encode the invariants (kswap never worse than backward,
  Jensen bound, UCB dominance, MILP-vs-bruteforce, hull-vertex recovery).

Habits that keep it on rails: ask for small steps and run the tests after
each; ask it to *explain the numbers it produces* (a wrong explanation
exposes a wrong run); when it proposes a new experiment, make it first say
which paper section / open problem the experiment answers; commit + push
often so the server and laptop never diverge.
